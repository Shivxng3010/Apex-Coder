import os
import sys
import json
import time
import hashlib
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple
from harness.schemas import Language
from harness.self_correct import AutoVerifyResult, extract_code_and_tests


class FileLock:
    """
    Cross-platform file lock using msvcrt on Windows and fcntl on Unix.
    Prevents concurrent write corruptions when /selfplay and chat write simultaneously.
    """
    def __init__(self, lock_file: Path, timeout: float = 5.0, poll_interval: float = 0.05):
        self.lock_file = Path(lock_file)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._fd = None

    def __enter__(self):
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        while True:
            try:
                self._fd = os.open(str(self.lock_file), os.O_RDWR | os.O_CREAT | os.O_TRUNC)
                if sys.platform == "win32":
                    import msvcrt
                    msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except (IOError, OSError):
                if self._fd is not None:
                    try:
                        os.close(self._fd)
                    except Exception:
                        pass
                    self._fd = None
                if time.time() - start_time >= self.timeout:
                    # Non-fatal timeout, return without blocking forever
                    return self
                time.sleep(self.poll_interval)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fd is not None:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None


class ExperienceHarvester:
    """
    Captures, filters, deduplicates, and stores 100% verified chat/selfplay experiences
    into the persistent live training pool and records DPO preference pairs (chosen vs rejected).
    """
    def __init__(self, pool_dir: str = "E:/apex-coder/data/experience_pool"):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.gold_file = self.pool_dir / "live_gold.jsonl"
        self.dpo_file = self.pool_dir / "live_dpo_pairs.jsonl"
        self.history_file = self.pool_dir / "consumed_history.jsonl"
        self.index_file = self.pool_dir / "hashes.json"
        self.lock_file = self.pool_dir / "experience.lock"

    def _load_hashes(self) -> set:
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    return set(json.load(f))
            except Exception:
                return set()
        return set()

    def _save_hashes(self, hashes: set):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(list(hashes), f)

    def harvest(
        self,
        user_query: str,
        response_text: str,
        verify_res: AutoVerifyResult,
        source: str = "interactive_chat",
    ) -> Tuple[bool, str]:
        """
        Validates, deduplicates, and logs a verified interaction into live_gold.jsonl.
        Returns (success: bool, reason_or_id: str).
        """
        # 1. Strict Verification Gate: only 100% passing tests are harvested
        if not verify_res.has_tests or not verify_res.is_valid:
            return False, "Sample did not pass test verification."

        impl_code = verify_res.extracted_code
        test_code = verify_res.extracted_test
        if not impl_code or not test_code:
            _, impl_code, test_code = extract_code_and_tests(response_text)

        if not impl_code or not test_code:
            return False, "Missing extracted implementation or test code."

        # 2. Quality Filter: Minimum code length & assertion check
        impl_lines = [l for l in impl_code.strip().splitlines() if l.strip()]
        test_lines = [l for l in test_code.strip().splitlines() if l.strip()]

        if len(impl_lines) < 3:
            return False, "Solution code too short / trivial."
        if len(test_lines) < 2:
            return False, "Test code too short / trivial."
        if "assert" not in test_code.lower() and "expect" not in test_code.lower():
            return False, "No assert or expect statements found in test code."

        # 3. Deduplication Hash
        query_norm = " ".join(user_query.strip().lower().split())
        content_hash = hashlib.sha256(f"{query_norm}|||{impl_code}".encode("utf-8")).hexdigest()

        with FileLock(self.lock_file):
            known_hashes = self._load_hashes()
            if content_hash in known_hashes:
                return False, "Duplicate sample (hash already recorded)."

            lang_str = verify_res.language.value if verify_res.language else "typescript"
            sample_id = f"exp_{int(time.time())}_{content_hash[:8]}"

            record = {
                "id": sample_id,
                "timestamp": time.time(),
                "source": source,
                "language": lang_str,
                "duration_sec": round(verify_res.duration_seconds, 3),
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Apex Coder, an expert systems engineer and coding assistant."
                    },
                    {
                        "role": "user",
                        "content": user_query
                    },
                    {
                        "role": "assistant",
                        "content": response_text
                    }
                ],
                "metadata": {
                    "impl_lines": len(impl_lines),
                    "test_lines": len(test_lines),
                    "hash": content_hash
                }
            }

            with open(self.gold_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            known_hashes.add(content_hash)
            self._save_hashes(known_hashes)

        return True, sample_id

    def harvest_dpo_pair(
        self,
        user_query: str,
        chosen_text: str,
        rejected_text: str,
        rejection_reason: str = "Test assertion or syntax failure",
        language: Optional[Language] = None,
        source: str = "self_correction_loop",
    ) -> Tuple[bool, str]:
        """
        Stores high-value preference optimization pairs (chosen vs rejected)
        for DPO / ORPO training to minimize model hallucinations.
        """
        if not chosen_text or not rejected_text or chosen_text == rejected_text:
            return False, "Invalid chosen/rejected pair"

        dpo_id = f"dpo_{int(time.time())}_{hashlib.sha256(user_query.encode('utf-8')).hexdigest()[:8]}"
        lang_str = language.value if language else "typescript"

        dpo_record = {
            "id": dpo_id,
            "timestamp": time.time(),
            "source": source,
            "language": lang_str,
            "prompt": user_query,
            "rejection_reason": rejection_reason,
            "chosen": chosen_text,
            "rejected": rejected_text,
            "messages_chosen": [
                {
                    "role": "system",
                    "content": "You are Apex Coder, an expert systems engineer and coding assistant."
                },
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": chosen_text}
            ],
            "messages_rejected": [
                {
                    "role": "system",
                    "content": "You are Apex Coder, an expert systems engineer and coding assistant."
                },
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": rejected_text}
            ]
        }

        with FileLock(self.lock_file):
            with open(self.dpo_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(dpo_record, ensure_ascii=False) + "\n")

        return True, dpo_id

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns stats about pending live gold samples, DPO preference pairs, and historical training.
        """
        with FileLock(self.lock_file):
            pending_count = 0
            if self.gold_file.exists():
                with open(self.gold_file, "r", encoding="utf-8") as f:
                    pending_count = sum(1 for line in f if line.strip())

            dpo_count = 0
            if self.dpo_file.exists():
                with open(self.dpo_file, "r", encoding="utf-8") as f:
                    dpo_count = sum(1 for line in f if line.strip())

            history_count = 0
            if self.history_file.exists():
                with open(self.history_file, "r", encoding="utf-8") as f:
                    history_count = sum(1 for line in f if line.strip())

            return {
                "pending_gold_samples": pending_count,
                "pending_dpo_pairs": dpo_count,
                "consumed_history_samples": history_count,
                "pool_dir": str(self.pool_dir.resolve()),
                "gold_file_exists": self.gold_file.exists(),
                "dpo_file_exists": self.dpo_file.exists(),
            }
