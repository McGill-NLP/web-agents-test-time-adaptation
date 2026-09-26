"""
Prompt builder for GenericAgent

It is based on the dynamic_prompting module from the agentlab package.
This modified version supports user-specific image folders (User01, User02, ...)
and prebuilds per-user embeddings + FAISS indexes (cached under ./cache/).
"""

import os
import base64
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
import logging
from dataclasses import dataclass
from typing import Dict, List

from browsergym.core import action
from browsergym.core.action.base import AbstractActionSet

from agentlab.agents import dynamic_prompting as dp
from agentlab.llm.llm_utils import HumanMessage, parse_html_tags_raise

import atexit

MODEL_NAME = "openai/gpt-5-nano"   

MODEL_PRICING = {
    "openai/gpt-5-nano": {
        "input_text_1k": 0.00005,
        "cached_input_1k": 0.000005,
        "output_text_1k": 0.0004,
        "input_image_1k": 0.00005,  
    }
}

TOTAL_COST_STATS = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cached_tokens": 0,
    "image_tokens": 0,
    "total_cost": 0.0
}

def save_cost_report(filepath: str = f"cost_report_{MODEL_NAME}.txt"):
    report = []
    report.append("==== COST REPORT ====\n\n")
    report.append(f"Model: {MODEL_NAME}\n\n")

    report.append(f"Total Prompt Tokens: {TOTAL_COST_STATS['prompt_tokens']}\n")
    report.append(f"  - Cached Tokens: {TOTAL_COST_STATS['cached_tokens']}\n")
    report.append(f"  - Image Tokens: {TOTAL_COST_STATS['image_tokens']}\n")

    pure_text = TOTAL_COST_STATS['prompt_tokens'] - TOTAL_COST_STATS['cached_tokens'] - TOTAL_COST_STATS['image_tokens']
    report.append(f"  - Pure Text Tokens: {pure_text}\n")

    report.append(f"\nTotal Completion Tokens: {TOTAL_COST_STATS['completion_tokens']}\n")
    report.append(f"\nTotal Cost (USD): ${TOTAL_COST_STATS['total_cost']:.6f}\n")

    report.append("\n==== PRICING ====\n")
    if MODEL_NAME in MODEL_PRICING:
        for k, v in MODEL_PRICING[MODEL_NAME].items():
            report.append(f"{k}: {v} per 1K tokens\n")

    with open(filepath, "w") as f:
        f.writelines(report)

    print(f"\n[INFO] Cost report saved to {filepath}")

atexit.register(save_cost_report)

#config
IMG_DIR = "User_History"
CACHE_ROOT = "./cache"
EMBED_CACHE_DIR = os.path.join(CACHE_ROOT, "embeddings")
FAISS_CACHE_DIR = os.path.join(CACHE_ROOT, "faiss")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 512 

# Ensure cache directories exist
os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
os.makedirs(FAISS_CACHE_DIR, exist_ok=True)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_MODEL = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
CLIP_PROCESSOR = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

# USER_IMAGE_PATHS: user_id -> list of image paths (consistent order with embeddings)
USER_IMAGE_PATHS: Dict[str, List[str]] = {}
# INDEXES: user_id -> faiss.Index (CPU index)
INDEXES: Dict[str, faiss.Index] = {}
# EMBEDDINGS_CACHE: user_id -> numpy array (n_images, dim)
EMBEDDINGS_CACHE: Dict[str, np.ndarray] = {}


def get_user_dirs(img_root: str) -> List[str]:
    """
    Returns a list of user_id folder names inside img_root.
    Accepts any directory name (we assume user folders are immediate children).
    """
    entries = []
    for name in sorted(os.listdir(img_root)):
        full = os.path.join(img_root, name)
        if os.path.isdir(full):
            entries.append(name)
    return entries

def collect_image_paths_for_user(user_id: str) -> List[str]:
    user_dir = os.path.join(IMG_DIR, user_id)
    if not os.path.isdir(user_dir):
        return []

    imgs = []
    for root, dirs, files in os.walk(user_dir):
        for fname in files:
            if fname.lower().endswith(".png"):
                imgs.append(os.path.join(root, fname))
    return sorted(imgs)

def embed_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    inputs = CLIP_PROCESSOR(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = CLIP_MODEL.get_image_features(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embedding = outputs.pooler_output
        elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            embedding = outputs.image_embeds
        else:
            embedding = outputs

    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    return embedding.cpu().numpy().astype("float32")


def build_embeddings_for_user(user_id: str) -> np.ndarray:
    """
    Build or load cached embeddings for given user_id.
    Returns numpy array of shape (n_images, EMBEDDING_DIM)
    """
    cache_path = os.path.join(EMBED_CACHE_DIR, f"{user_id}_embeddings.npy")
    image_paths = collect_image_paths_for_user(user_id)
    if len(image_paths) == 0:
        logging.warning(f"No images found for user '{user_id}' in {os.path.join(IMG_DIR, user_id)}")
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")

    if os.path.exists(cache_path):
        try:
            embs = np.load(cache_path)
            if embs.shape[1] == EMBEDDING_DIM and embs.shape[0] == len(image_paths):
                logging.info(f"Loaded cached embeddings for {user_id} ({embs.shape[0]} images).")
                return embs
            else:
                logging.info(f"Embedding cache shape mismatch for {user_id}, recomputing.")
        except Exception as e:
            logging.warning(f"Failed to load embedding cache for {user_id}: {e}. Recomputing.")

    # compute embeddings
    all_embs = []
    for p in image_paths:
        try:
            emb = embed_image(p)
            all_embs.append(emb[0])
        except Exception as e:
            logging.exception(f"Failed to embed image {p}: {e}")

    embs = np.vstack(all_embs).astype("float32")
    # save cache
    np.save(cache_path, embs)
    logging.info(f"Saved embeddings cache for {user_id} -> {cache_path}")
    return embs


def build_faiss_index_for_user(user_id: str, embeddings: np.ndarray) -> faiss.Index:
    """
    Build or load cached FAISS index for given user_id.
    Uses IndexFlatIP for cosine-like similarity (inputs should be normalized).
    Returns a CPU faiss.Index.
    """
    index_path = os.path.join(FAISS_CACHE_DIR, f"{user_id}.index")
    if os.path.exists(index_path):
        try:
            idx = faiss.read_index(index_path)
            logging.info(f"Loaded FAISS index from cache for {user_id}.")
            return idx
        except Exception as e:
            logging.warning(f"Failed to load FAISS index for {user_id}: {e}. Rebuilding.")

    # Normalize embeddings to unit length (so IP ~ cosine)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms

    idx = faiss.IndexFlatIP(EMBEDDING_DIM)
    if normalized.shape[0] > 0:
        idx.add(normalized)
    # persist to disk
    try:
        faiss.write_index(idx, index_path)
        logging.info(f"Saved FAISS index cache for {user_id} -> {index_path}")
    except Exception as e:
        logging.warning(f"Could not write FAISS index cache for {user_id}: {e}")

    return idx


def prepare_all_user_indexes():
    """
    Scan IMG_DIR, build/load embeddings and indexes for each user folder.
    Populates USER_IMAGE_PATHS, EMBEDDINGS_CACHE, INDEXES.
    """
    user_dirs = get_user_dirs(IMG_DIR)
    if not user_dirs:
        logging.warning(f"No user subfolders found under {IMG_DIR}.")
    for uid in user_dirs:
        img_paths = collect_image_paths_for_user(uid)
        USER_IMAGE_PATHS[uid] = img_paths
        if len(img_paths) == 0:
            EMBEDDINGS_CACHE[uid] = np.zeros((0, EMBEDDING_DIM), dtype="float32")
            INDEXES[uid] = faiss.IndexFlatIP(EMBEDDING_DIM)
            continue
        embs = build_embeddings_for_user(uid)
        EMBEDDINGS_CACHE[uid] = embs
        idx = build_faiss_index_for_user(uid, embs)
        INDEXES[uid] = idx
    logging.info(f"Prepared indexes for users: {list(INDEXES.keys())}")


# Immediately prepare at module import/startup
prepare_all_user_indexes()

def _compute_text_embedding(text: str) -> np.ndarray:
    text_inputs = CLIP_PROCESSOR(text=[text], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_features = CLIP_MODEL.get_text_features(**text_inputs)
    
    if hasattr(text_features, "pooler_output"):
        text_features = text_features.pooler_output
    elif hasattr(text_features, "last_hidden_state"):
        text_features = text_features.last_hidden_state[:, 0, :]
        
    text_embedding = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_embedding.cpu().numpy().astype("float32")

# Config
TOP_K = 2  # number of trajectories to retrieve

def retrieve_and_summarize(task_instruction: str, user_id: str) -> str:

    if user_id not in INDEXES:
        raise ValueError(f"user_id '{user_id}' not found.")

    text_emb = _compute_text_embedding(task_instruction)
    idx = INDEXES[user_id]

    if idx.ntotal == 0:
        return ""

    image_paths = USER_IMAGE_PATHS[user_id]
    all_indices = list(range(len(image_paths)))

    selected_trajectories = []

    def extract_step_from_path(path):
        fname = os.path.basename(path)
        try:
            return int(fname.split("screenshot_step_")[-1].split("_")[0])
        except:
            return None

    def extract_step(fname: str):
        try:
            return int(fname.split("screenshot_step_")[-1].split("_")[0])
        except:
            return 10**9

    # Sequential retrieval
    remaining_indices = np.array(all_indices)

    for k in range(TOP_K):

        if len(remaining_indices) == 0:
            break

        sub_embeddings = EMBEDDINGS_CACHE[user_id][remaining_indices]

        temp_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        temp_index.add(sub_embeddings)

        D, I = temp_index.search(text_emb, 1)

        local_idx = int(I[0][0])
        global_idx = int(remaining_indices[local_idx])

        best_path = image_paths[global_idx]
        traj_folder = os.path.dirname(best_path)
        step = extract_step_from_path(best_path)

        print(f"\n[DEBUG] Selected traj {k+1}: {traj_folder}, step={step}")

        selected_trajectories.append((traj_folder, step))

        new_remaining = []
        for idx_ in remaining_indices:
            p = image_paths[int(idx_)]
            if os.path.dirname(p) != traj_folder:
                new_remaining.append(idx_)

        remaining_indices = np.array(new_remaining)

    all_images = []

    for traj_folder, max_step in selected_trajectories:

        for fname in sorted(os.listdir(traj_folder), key=extract_step):
            if not fname.startswith("screenshot_step_"):
                continue

            s = extract_step(fname)

            if max_step is None or s <= max_step:
                all_images.append(os.path.join(traj_folder, fname))

    print("\n[DEBUG] Total images used:", len(all_images))

    image_contents = []

    for path in all_images:
        try:
            with open(path, "rb") as f:
                b = base64.b64encode(f.read()).decode("utf-8")

            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b}"}
            })

        except Exception as e:
            logging.exception(f"Failed: {path}")

    # LLM summarization
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    response = client.chat.completions.create(
        model= MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert assistant skilled in inferring user preferences from past interaction trajectories. "
                    "You will receive a sequence of images representing a user’s past interactions with interfaces. "
                    
                    "These images may come from one or multiple interaction trajectories. "
                    "Carefully analyze only the visible actions and decisions shown in the images to infer the user’s preferences. "
                    
                    "Do not speculate or assume anything that is not directly supported by the visual evidence. "
                    
                    "If the interactions reflect a single consistent pattern, describe it clearly. "
                    "If multiple distinct or complementary patterns are present, integrate them into a coherent preference description. "
                    
                    "Provide a grounded, evidence-based summary written in the first person (e.g., 'I prefer...', 'I tend to...') "
                    "as a single concise sentence."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "From these interaction sequences, summarize my preferences. "
                            "Use first-person narration starting with 'I'."
                        ),
                    },
                    *image_contents,
                ],
            },
        ],
    )

    usage = response.usage

    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    details = getattr(usage, "prompt_tokens_details", None)

    cached_tokens = 0
    image_tokens = 0

    if details:
        cached_tokens = getattr(details, "cached_tokens", 0)
        image_tokens = getattr(details, "image_tokens", 0)

    text_tokens = prompt_tokens - cached_tokens - image_tokens

    rates = MODEL_PRICING[MODEL_NAME]

    cost = (
        text_tokens / 1000 * rates["input_text_1k"] +
        cached_tokens / 1000 * rates["cached_input_1k"] +
        image_tokens / 1000 * rates["input_image_1k"] +
        completion_tokens / 1000 * rates["output_text_1k"]
    )

    TOTAL_COST_STATS["prompt_tokens"] += prompt_tokens
    TOTAL_COST_STATS["completion_tokens"] += completion_tokens
    TOTAL_COST_STATS["cached_tokens"] += cached_tokens
    TOTAL_COST_STATS["image_tokens"] += image_tokens
    TOTAL_COST_STATS["total_cost"] += cost

    print("\n[COST DEBUG]")
    print(f"Text tokens: {text_tokens}")
    print(f"Cached tokens: {cached_tokens}")
    print(f"Image tokens: {image_tokens}")
    print(f"Output tokens: {completion_tokens}")
    print(f"Call Cost: ${cost:.6f}")
    print(f"Total Cost: ${TOTAL_COST_STATS['total_cost']:.6f}")

    return response.choices[0].message.content

# -----------------------------
# Prompt elements & MainPrompt (unchanged logic mostly) but passing user_id
# -----------------------------
@dataclass
class GenericPromptFlags(dp.Flags):
    obs: dp.ObsFlags
    action: dp.ActionFlags
    use_plan: bool = False
    use_criticise: bool = False
    use_thinking: bool = False
    use_memory: bool = False
    use_concrete_example: bool = True
    use_abstract_example: bool = False
    use_hints: bool = False
    enable_chat: bool = False
    max_prompt_tokens: int = None
    be_cautious: bool = True
    extra_instructions: str | None = None
    add_missparsed_messages: bool = True
    max_trunc_itr: int = 20
    flag_group: str = None


class ImageSummary(dp.PromptElement):
    def __init__(self, summary: str, visible: bool = True):
        super().__init__(visible=visible)
        self._prompt = f"<image_summary>\n{summary}\n</image_summary>"

    def _parse_answer(self, text_answer):
        return {}


class Memory(dp.PromptElement):
    _prompt = ""
    _abstract_ex = """
<memory>
Write down anything you need to remember for next steps. ...
</memory>
"""
    _concrete_ex = """
<memory>
I clicked on bid "32" to activate tab 2...
</memory>
"""
    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["memory"], merge_multiple=True)


class Plan(dp.PromptElement):
    def __init__(self, previous_plan, plan_step, visible: bool = True) -> None:
        super().__init__(visible=visible)
        self.previous_plan = previous_plan
        self._prompt = f"""
# Plan:

You just executed step {plan_step} of the previously proposed plan:\n{previous_plan}\n
After reviewing the effect of your previous actions, verify if your plan is still
relevant and update it if necessary.
"""

    _abstract_ex = """
<plan> ... </plan>
"""
    _concrete_ex = """
<plan> ... </plan>
"""
    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["plan", "step"])


class Criticise(dp.PromptElement):
    _prompt = ""
    _abstract_ex = """
<action_draft>... </action_draft>
"""
    _concrete_ex = """
<action_draft>... </action_draft>
"""
    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["action_draft", "criticise"])


class MainPrompt(dp.Shrinkable):
    def __init__(
        self,
        action_set: AbstractActionSet,
        obs_history: list[dict],
        actions: list[str],
        memories: list[str],
        thoughts: list[str],
        previous_plan: str,
        step: int,
        flags: dp.Flags,
    ) -> None:
        super().__init__()
        self.flags = flags
        self.history = dp.History(obs_history, actions, memories, thoughts, flags.obs)

        if self.flags.enable_chat:
            self.instructions = dp.ChatInstructions(
                obs_history[-1]["chat_messages"], extra_instructions=flags.extra_instructions
            )
        else:
            if sum([msg["role"] == "user" for msg in obs_history[-1].get("chat_messages", [])]) > 1:
                logging.warning("Agent is in goal mode, but multiple user messages exist. Consider enable_chat=True.")
            self.instructions = dp.GoalInstructions(
                obs_history[-1]["goal_object"], extra_instructions=flags.extra_instructions
            )

        self.obs = dp.Observation(obs_history[-1], self.flags.obs)
        self.action_prompt = dp.ActionPrompt(action_set, action_flags=flags.action)

        def time_for_caution():
            return flags.be_cautious and (flags.action.action_set.multiaction or flags.action.action_set == "python")

        self.be_cautious = dp.BeCautious(visible=time_for_caution)
        self.think = dp.Think(visible=lambda: flags.use_thinking)
        self.hints = dp.Hints(visible=lambda: flags.use_hints)
        self.plan = Plan(previous_plan, step, lambda: flags.use_plan)
        self.criticise = Criticise(visible=lambda: flags.use_criticise)
        self.memory = Memory(visible=lambda: flags.use_memory)

        # image summary
        task_instruction = obs_history[-1]["goal_object"]
        # read user_id from obs_history (requirement)
        text = obs_history[-1]["goal_object"][0]['text']
        user_id = text.split(":")[0].strip()
        text_filtered = text.split(":")[1].strip()
        print("\n[DEBUG] TEXT FILTERED:", text_filtered)
        if user_id is None:
            # be explicit: for safety, require user_id
            raise ValueError("obs_history[-1] must include 'user_id' to perform user-specific retrieval.")
        image_summary_text = retrieve_and_summarize(text_filtered, user_id=user_id)

        print("\n[DEBUG] Retrieved Image Summary Text:")
        print(image_summary_text)
        print("=" * 80)
        self.image_summary = ImageSummary(image_summary_text, visible=True)

    @property
    def _prompt(self) -> HumanMessage:
        prompt = HumanMessage(self.instructions.prompt)
        prompt.add_text(
            f"{self.image_summary.prompt}"
            f"{self.obs.prompt}{self.history.prompt}{self.action_prompt.prompt}{self.hints.prompt}"
            f"{self.be_cautious.prompt}{self.think.prompt}{self.plan.prompt}{self.memory.prompt}"
            f"{self.criticise.prompt}"
        )

        print("\n[DEBUG] Image Summary Section:")
        print(self.image_summary.prompt)
        print("=" * 80)

        return self.obs.add_screenshot(prompt)

    def shrink(self):
        self.history.shrink()
        self.obs.shrink()

    def _parse_answer(self, text_answer):
        ans_dict = {}
        ans_dict.update(self.think.parse_answer(text_answer))
        ans_dict.update(self.plan.parse_answer(text_answer))
        ans_dict.update(self.memory.parse_answer(text_answer))
        ans_dict.update(self.criticise.parse_answer(text_answer))
        ans_dict.update(self.action_prompt.parse_answer(text_answer))
        return ans_dict
