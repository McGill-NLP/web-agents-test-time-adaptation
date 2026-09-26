from dataclasses import dataclass
from agentlab.experiments.loop import ExpArgs
import os
from pathlib import Path

@dataclass
class WebSynthExpArgs(ExpArgs):
    def run(self):
        # ✅ 절대경로 강제 복구
        exp_root = Path(os.environ["AGENTLAB_EXP_ROOT"]).resolve()
        os.environ["AGENTLAB_EXP_ROOT"] = str(exp_root)

        print(f"[Worker {os.getpid()}] EXP_ROOT =", os.environ["AGENTLAB_EXP_ROOT"])

        import websynth.register

        return super().run()