"""
To run:

```bash
export SUFFIX="xl-0"

source vars/set_cf_vars.sh
python run_agent.py -b "mini_webarena" -m "gpt-4o-mini"
"""
import argparse
import logging
from pathlib import Path
import os
from dataclasses import dataclass
from agentlab.experiments import study as _study_mod
from agentlab.experiments.websynth_loop import WebSynthExpArgs

_study_mod.ExpArgs = WebSynthExpArgs  # 🔥 핵심

# this needs to be set before importing agentlab
#default_exp_root = str(Path(__file__).parent / "agentlab_results")
#default_exp_root = str(Path("/network/scratch/s/shind") / "agentlab_results")
#os.environ["AGENTLAB_EXP_ROOT"] = os.getenv("AGENTLAB_EXP_ROOT", default_exp_root)
#print("AGENTLAB_EXP_ROOT:", os.environ["AGENTLAB_EXP_ROOT"])

# $SCRATCH 환경 변수를 사용하여 경로 설정
# scratch_dir = os.environ.get("SCRATCH", "/network/scratch/s/shind")  # 기본값은 /network/scratch/s/shind
# default_exp_root = str(Path(scratch_dir) / "agentlab_results")
from pathlib import Path
import os

scratch_dir = Path(os.environ.get("SCRATCH", "/network/scratch/s/shind")).resolve()
default_exp_root = (scratch_dir / "agentlab_results").resolve()

# 문자열로 넣기 (중요)
os.environ["AGENTLAB_EXP_ROOT"] = str(default_exp_root)

# 폴더 생성
default_exp_root.mkdir(parents=True, exist_ok=True)

# 디버깅 출력 (강추)
print("AGENTLAB_EXP_ROOT =", os.environ["AGENTLAB_EXP_ROOT"])

from agentlab.experiments.study import Study
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.chat_api import ChatModel
from openai import OpenAI

class OpenAICompatibleChatModel(ChatModel):
    def __init__(
        self, 
        model_name,
        api_key_env_var,
        base_url_env_var,
        api_key=None,
        base_url=None,
        temperature=1,
        max_completion_tokens=1024,
        max_retry=4,
        min_retry_wait_time=60,
    ):
        import agentlab.llm.tracking as tracking

        if not api_key_env_var in os.environ:
            raise ValueError(f"{api_key_env_var} must be set in the environment")
        if not base_url_env_var in os.environ:
            raise ValueError(f"{base_url_env_var} must be set in the environment")

        if base_url is None:
            base_url = os.environ[base_url_env_var]
        
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_completion_tokens,  # 수정
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            api_key_env_var=api_key_env_var,
            client_class=OpenAI,
            client_args={
                "base_url": base_url,
            },
            pricing_func=tracking.get_pricing_openai,
        )

@dataclass
class OpenRouterModelArgs(BaseModelArgs):
    def make_model(self):
        return OpenAICompatibleChatModel(
            model_name=self.model_name,
            temperature=1,
            max_completion_tokens=self.max_new_tokens,
            api_key_env_var="OPENAI_API_KEY",
            base_url_env_var="OPENAI_BASE_URL",
        )

@dataclass
class VllmModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    def set_base_url(self, base_url):
        self.base_url = base_url

    def set_api_key(self, api_key):
        self.api_key = api_key
    
    def make_model(self):
        base_url = None if not hasattr(self, "base_url") else self.base_url
        api_key = None if not hasattr(self, "api_key") else self.api_key
        
        return OpenAICompatibleChatModel(
            model_name=self.model_name,
            temperature=1,
            max_completion_tokens=self.max_new_tokens,
            api_key_env_var="VLLM_API_KEY",
            base_url_env_var="VLLM_BASE_URL",
            base_url=base_url,
            api_key=api_key,
        )

@dataclass
class GeminiModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    def set_base_url(self, base_url):
        self.base_url = base_url

    def set_api_key(self, api_key):
        self.api_key = api_key
    
    def make_model(self):
        base_url = None if not hasattr(self, "base_url") else self.base_url
        api_key = None if not hasattr(self, "api_key") else self.api_key
        
        return OpenAICompatibleChatModel(
            model_name=self.model_name,
            temperature=1,
            max_completion_tokens=self.max_new_tokens,
            api_key_env_var="GEMINI_API_KEY",
            base_url_env_var="GEMINI_BASE_URL",
            base_url=base_url,
            api_key=api_key,
        )

def get_webarena_benchmark(split='train'):
    if split not in ['train', 'valid', 'test']:
        raise ValueError(f"split must be one of ['train', 'valid', 'test'], got {split}")

    from browsergym.experiments.benchmark.base import Benchmark
    from browsergym.experiments.benchmark.configs import DEFAULT_HIGHLEVEL_ACTION_SET_ARGS
    from browsergym.experiments.benchmark.utils import make_env_args_list_from_fixed_seeds
    from browsergym.experiments.benchmark.metadata.utils import task_metadata, task_list_from_metadata
    
    tm = task_metadata("webarena")

    b = Benchmark(
        name=f"webarena_{split}",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["webarena"],
        is_multi_tab=True,
        supports_parallel_seeds=False,
        backends=["webarena"],
        env_args_list=make_env_args_list_from_fixed_seeds(
            task_list=task_list_from_metadata(tm),
            max_steps=30,
            fixed_seeds=[0],
        ),
        task_metadata=tm,
    )

    b_split = b.subset_from_split(split)

    return b_split


def get_websynth_benchmark(reset_instance=True):
    from browsergym.experiments.benchmark.base import Benchmark
    from browsergym.experiments.benchmark.configs import DEFAULT_HIGHLEVEL_ACTION_SET_ARGS
    # from browsergym.experiments.benchmark.utils import make_env_args_list_from_fixed_seeds
    from browsergym.experiments.benchmark.metadata.utils import task_metadata, task_list_from_metadata
    
    from websynth import get_task_ids, WebSynthBenchmark, CONFIG_PATH, modified_make_env_args_list_from_fixed_seeds

    task_ids = get_task_ids(CONFIG_PATH)

    b = WebSynthBenchmark(
        name="websynth",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["webarena"],
        is_multi_tab=True,
        supports_parallel_seeds=False,
        backends=[],
        env_args_list=modified_make_env_args_list_from_fixed_seeds(
            task_list=[f"websynth.{i}" for i in task_ids],
            max_steps=30,
            fixed_seeds=[0],
        ),
        task_metadata=None,
        reset_instance=reset_instance,
    )

    return b


def get_default_flags(
    use_screenshot=True,
    use_som=True,
    max_prompt_tokens=16384 - 4096,
    enable_chat=False,
):

    from agentlab.agents import dynamic_prompting as dp
    from agentlab.agents.generic_agent.generic_agent import GenericPromptFlags
    from browsergym.experiments.benchmark import HighLevelActionSetArgs

    flags = GenericPromptFlags(
        obs=dp.ObsFlags(
            use_html=False,
            use_ax_tree=True, #False
            use_focused_element=True,
            use_error_logs=True,
            use_history=True,
            use_past_error_logs=False,
            use_action_history=True,
            use_think_history=False,
            use_diff=False,
            html_type="pruned_html",
            use_screenshot=use_screenshot,
            use_som=use_som,
            extract_visible_tag=True,
            extract_clickable_tag=True,
            extract_coords="False",
            filter_visible_elements_only=False,
        ),
        action=dp.ActionFlags(
            action_set=HighLevelActionSetArgs(
                subsets=["bid"],
                multiaction=False,
            ),
            long_description=False,
            individual_examples=False,
        ),
        use_plan=False,
        use_criticise=False,
        use_thinking=False, #qwen instruct
        use_memory=False,
        use_concrete_example=True,
        use_abstract_example=True,
        use_hints=True,
        enable_chat=enable_chat,
        max_prompt_tokens=max_prompt_tokens,
        be_cautious=True,
        extra_instructions=None,
    )

    return flags

def prepare_vllm_model(
    model_name="Qwen/Qwen3-VL-30B-A3B-Thinking",
    max_new_tokens=1024,
    max_prompt_tokens=13000 - 4096,   # Qwen3 VL supports large context; adjust as needed
    max_total_tokens=13000,
    use_vision=True,
    enable_chat=False,
    base_url=None,
    api_key=None,
):
    # The base_url and api_key are passed to VllmModelArgs.make_model internally if needed.

    model_args = VllmModelArgs(
        model_name=model_name,
        max_total_tokens=max_total_tokens,
        max_input_tokens=max_total_tokens - max_new_tokens,
        max_new_tokens=max_new_tokens,
        vision_support=use_vision,
    )

    if base_url is not None:
        model_args.set_base_url(base_url)

    if api_key is not None:
        model_args.set_api_key(api_key)

    agent_args = GenericAgentArgs(
        chat_model_args=model_args,
        flags=get_default_flags(
            max_prompt_tokens=max_prompt_tokens,
            use_som=use_vision,
            use_screenshot=use_vision,
            enable_chat=enable_chat,
        ),
    )

    return agent_args

def prepare_gpt(
    model_name="gpt-5",
    max_new_tokens=1024,
    max_prompt_tokens=16384 - 4096,   # gpt-5-mini supports up to ~128k context
    max_total_tokens=16384,
    use_vision=True,
    use_som=True,
    enable_chat=False,
):
    from agentlab.llm.chat_api import OpenAIModelArgs

    agent_arg = GenericAgentArgs(
        chat_model_args=OpenAIModelArgs(
            model_name=model_name,
            max_total_tokens=max_total_tokens,
            max_input_tokens=max_total_tokens - max_new_tokens,
            #max_new_tokens=max_new_tokens,
            max_completion_tokens=max_new_tokens,  # 수정
            vision_support=use_vision,
        ),
        flags=get_default_flags(
            max_prompt_tokens=max_prompt_tokens,
            use_screenshot=use_vision,
            use_som=use_som,
            enable_chat=enable_chat,
        ),
    )

    return agent_arg

def prepare_gemini(
    model_name="gemini-3-pro-preview",
    max_new_tokens=10000,
    max_prompt_tokens=50000,
    max_total_tokens=60000,
    use_vision=False,
    enable_chat=False,
):
    model_args = GeminiModelArgs(
        model_name=model_name,
        max_total_tokens=max_total_tokens,
        max_input_tokens=max_total_tokens - max_new_tokens,
        max_new_tokens=max_new_tokens,
        vision_support=use_vision,
    )

    agent_args = GenericAgentArgs(
        chat_model_args=model_args,
        flags=get_default_flags(
            max_prompt_tokens=max_prompt_tokens,
            use_screenshot=use_vision,
            use_som=use_vision,
            enable_chat=enable_chat,
        ),
    )

    return agent_args

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser(
    description="Run a generic agent on a benchmark",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "-r",
    "--relaunch",
    type=str,
    help="Relaunch an existing study with a string that matches the study name",
)

parser.add_argument(
    "-b",
    "--benchmark",
    choices=[
        "webarena_train",
        "custom",
        "websynth",
        "webarena_test",
    ],
    default="websynth",
    help="Select the benchmark to run on",
)

parser.add_argument(
    "-m",
    "--models",
    "--model",
    choices=[
        "gpt-5",
        "gpt-5.2",
        "gpt-5.4",
        "gpt-5-mini",
        "qwen3-vl-30b-a3b-thinking",
        "qwen3-vl-32b-instruct", 
        "qwen3-vl-32b-thinking", 
        "qwen3-vl-8b-thinking", 
        "qwen3.5-27b",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview"  
    ],
    default=["gemini-3-pro-preview"],
    help="Select the model to use",
    # allow multiple models, but at least one
    nargs="+",
)
parser.add_argument(
    "-n",
    "--n_jobs",
    type=int,
    default=4,
    help="Number of parallel jobs to run",
)
parser.add_argument(
    "--parallel",
    choices=["ray", "joblib", "sequential"],
    default="ray",
    help="Select the parallel backend to use",
)
parser.add_argument(
    "--disable-instance-reset",
    action="store_true",
    help="Disable the instance reset for the websynth benchmark",
)
args = parser.parse_args()

if __name__ == "__main__":  # necessary for dask backend
    # 1. select benchmark:
    if args.benchmark == "webarena_train":
        benchmark = get_webarena_benchmark(split='train')
    if args.benchmark == "webarena_test":
        benchmark = get_webarena_benchmark(split='test')
    elif args.benchmark == "websynth":
        reset_instance = not args.disable_instance_reset
        benchmark = get_websynth_benchmark(reset_instance=reset_instance)
    elif args.benchmark == "custom":
        # this is not implemented yet
        raise NotImplementedError("Custom benchmark is not implemented yet")
    else:
        raise ValueError(f"Unknown benchmark {args.benchmark}")

    # 2. Select model
    # since args.models is a list now, we'll make a list of agent_args
    agent_args = []
    for model in args.models:
        if model == "gpt-5":
            # must set OPENAI_API_KEY in environment
            agent_args.append(prepare_gpt(model_name='gpt-5'))
        elif model == "gpt-5.2":
            # must set OPENAI_API_KEY in environment
            agent_args.append(prepare_gpt(model_name='gpt-5.2'))
        elif model == "gpt-5.4":
            # must set OPENAI_API_KEY in environment
            agent_args.append(prepare_gpt(model_name='gpt-5.4'))
        elif model == "gpt-5-mini":
            # must set OPENAI_API_KEY in environment
            agent_args.append(prepare_gpt(model_name='gpt-5-mini'))
        elif model == "qwen3-vl-32b-instruct":
            # must set VLLM_API_KEY and VLLM_BASE_URL in environment
            agent_args.append(prepare_vllm_model(
                model_name="Qwen/Qwen3-VL-32B-Instruct",
                use_vision=False,
            ))
        elif model == "qwen3-vl-30b-a3b-thinking":
            # must set VLLM_API_KEY and VLLM_BASE_URL in environment
            agent_args.append(prepare_vllm_model(
                model_name="Qwen/Qwen3-VL-30B-A3B-Thinking",
                use_vision=False,
            ))
        elif model == "qwen3-vl-32b-thinking":
            # must set VLLM_API_KEY and VLLM_BASE_URL in environment
            agent_args.append(prepare_vllm_model(
                model_name="Qwen/Qwen3-VL-32B-Thinking",
                use_vision=False,
            ))
        elif model == "qwen3-vl-8b-thinking":
            # must set VLLM_API_KEY and VLLM_BASE_URL in environment
            agent_args.append(prepare_vllm_model(
                model_name="Qwen/Qwen3-VL-8B-Thinking",
                use_vision=False,
            ))
        elif model == "qwen3.5-27b":
            agent_args.append(prepare_vllm_model(
                model_name="Qwen/Qwen3.5-27B",
                use_vision=False,
            ))
        elif model == "gemini-3-pro-preview":
            # must set GEMINI_API_KEY and GEMINI_BASE_URL in environment
            agent_args.append(prepare_gemini(
                model_name="gemini-3-pro-preview", #openrouter name
                #use_vision=False,
            ))
        elif model == "gemini-3-flash-preview":
            # must set GEMINI_API_KEY and GEMINI_BASE_URL in environment
            agent_args.append(prepare_gemini(
                model_name="gemini-3-flash-preview",
                #use_vision=False,
            ))
        else:
            raise ValueError(f"Unknown model {model}")


    # 3. Set up study
    n_relaunch = 3
    parallel_backend = args.parallel
    n_jobs = args.n_jobs

    reproducibility_mode = True
    strict_reproducibility = False

    if reproducibility_mode:
        [a.set_reproducibility_mode() for a in agent_args]

    if args.relaunch is not None:
        print("Relaunching study from directory containing:", args.relaunch)
        study = Study.load_most_recent(contains=args.relaunch, root_dir=Path(os.environ["AGENTLAB_EXP_ROOT"]))
        study.find_incomplete(include_errors=True)
    else:
        study = Study(agent_args, benchmark, logging_level_stdout=logging.INFO)  # type: ignore

    # 4. Run study
    study.run(
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        strict_reproducibility=strict_reproducibility,
        n_relaunch=n_relaunch,
        exp_root=Path(os.environ["AGENTLAB_EXP_ROOT"]),  # 🔥 핵심
    )

    if reproducibility_mode:
        study.append_to_journal(strict_reproducibility=strict_reproducibility)