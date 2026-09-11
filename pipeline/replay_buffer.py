import os
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from harness.experience_harvester import FileLock


class ReplayBuffer:
    """
    Blends new live experience gold data with historical anchor data
    in an 80/20 ratio to prevent catastrophic forgetting during continual fine-tuning.
    """
    def __init__(
        self,
        pool_dir: str = "E:/apex-coder/data/experience_pool",
        anchor_datasets: Optional[List[str]] = None,
    ):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.gold_file = self.pool_dir / "live_gold.jsonl"
        self.history_file = self.pool_dir / "consumed_history.jsonl"
        self.staged_file = self.pool_dir / "staged_batch.jsonl"
        self.lock_file = self.pool_dir / "experience.lock"

        if anchor_datasets is None:
            self.anchor_paths = [
                Path("E:/apex-coder/data/clean/train_2k.jsonl"),
                Path("E:/apex-coder/data/clean/concurrency_300.jsonl"),
            ]
        else:
            self.anchor_paths = [Path(p) for p in anchor_datasets]

    def _load_anchor_samples(self, count: int) -> List[Dict[str, Any]]:
        pool = []
        for path in self.anchor_paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                pool.append(json.loads(line))
                            except Exception:
                                pass

        if not pool:
            return []

        if len(pool) <= count:
            return pool
        return random.sample(pool, count)

    def prepare_staged_batch(self, min_samples: int = 1, live_ratio: float = 0.8) -> Optional[Path]:
        """
        Reads pending live samples, mixes with historical anchor samples (80/20 blend),
        and writes the staged training file. Returns the path to staged_batch.jsonl.
        """
        with FileLock(self.lock_file):
            if not self.gold_file.exists():
                return None

            live_samples = []
            with open(self.gold_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            live_samples.append(json.loads(line))
                        except Exception:
                            pass

            if len(live_samples) < min_samples:
                return None

            # Calculate anchor count for 80/20 ratio: anchor / (live + anchor) = 0.20 => anchor = live * 0.25
            live_count = len(live_samples)
            anchor_count = max(2, int(live_count * (1.0 - live_ratio) / live_ratio))
            anchor_samples = self._load_anchor_samples(anchor_count)

            # Combine and shuffle
            combined = live_samples + anchor_samples
            random.shuffle(combined)

            with open(self.staged_file, "w", encoding="utf-8") as f:
                for s in combined:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

            return self.staged_file

    def archive_consumed_batch(self):
        """
        After successful training and quality gate verification, moves pending live_gold
        records to consumed_history.jsonl and clears live_gold.jsonl.
        """
        with FileLock(self.lock_file):
            if not self.gold_file.exists():
                return

            with open(self.gold_file, "r", encoding="utf-8") as f_in:
                lines = f_in.readlines()

            if lines:
                with open(self.history_file, "a", encoding="utf-8") as f_out:
                    f_out.writelines(lines)

            # Clear live_gold.jsonl
            with open(self.gold_file, "w", encoding="utf-8") as f_clear:
                pass
