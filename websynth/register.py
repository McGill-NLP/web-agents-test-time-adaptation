import os
import importlib.resources
import logging
import gymnasium as gym
from pathlib import Path

import nltk
from browsergym.core.registration import register_task

from . import GenericWebSynthTask, get_task_ids, CONFIG_PATH

logger = logging.getLogger(__name__)

# download necessary tokenizer resources
# note: deprecated punkt -> punkt_tab https://github.com/nltk/nltk/issues/3293
try:
    nltk.data.find("tokenizers/punkt_tab")
except:
    nltk.download("punkt_tab", quiet=True, raise_on_error=True)

ALL_WEBSYNTH_GYM_IDS = []
ALL_WEBSYNTH_TASK_IDS = []

config_path: Path = CONFIG_PATH

# register all WebArena benchmark
for task_id in get_task_ids(config_path):
    gym_id = f"websynth.{task_id}"
    register_task(
        gym_id,
        GenericWebSynthTask,
        task_kwargs={"task_id": task_id, 'config_path': config_path},
    )
    ALL_WEBSYNTH_GYM_IDS.append(gym_id)
    ALL_WEBSYNTH_TASK_IDS.append(task_id)

logger.info(f"Registered {len(ALL_WEBSYNTH_GYM_IDS)} WebSynth tasks")
logger.info(f"Sample Task IDs: {ALL_WEBSYNTH_GYM_IDS[0:200:20]}")
print("[DEBUG] Registered WebSynth environments:")
print([env_id for env_id in gym.envs.registry.keys() if "websynth" in env_id])