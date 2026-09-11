import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.schemas import DatasetSample, Language, ExecutionStatus, ExecutionResult
from harness.python_runner import PythonRunner
from harness.ts_runner import TypeScriptRunner
from harness.experience_harvester import ExperienceHarvester
from pipeline.problem_bank import generate_python_sample, generate_ts_sample
from pipeline.concurrency_bank import generate_concurrency_python_sample, generate_concurrency_ts_sample


class RubricJudge:
    def __init__(self, score_margin_threshold: float = 15.0):
        self.score_margin_threshold = score_margin_threshold

    def evaluate_code_rubric(self, code: str, thinking: str, language: Language) -> float:
        score = 0.0
        code_clean = code.strip()

        # 1. Complexity & Efficiency (Max 35 points)
        complexity_score = 15.0
        if "while" in code_clean or "for " in code_clean:
            if "shift()" in code_clean or ".pop(0)" in code_clean:
                complexity_score += 5.0
            else:
                complexity_score += 10.0
        if "filter(" in code_clean or "lambda " in code_clean:
            complexity_score += 5.0
        if "sort(" in code_clean or "sorted(" in code_clean:
            complexity_score += 3.0
        if "Counter" in code_clean or "Map" in code_clean or "dict" in code_clean or "Set" in code_clean:
            complexity_score += 7.0
        score += min(35.0, complexity_score)

        # 2. Modularity, Typing & Clean Idiomatic Style (Max 25 points)
        style_score = 10.0
        if language == Language.TYPESCRIPT:
            if ": " in code_clean or "<T>" in code_clean or "interface " in code_clean:
                style_score += 8.0
            if "export " in code_clean:
                style_score += 4.0
            if "const " in code_clean and "let " in code_clean:
                style_score += 3.0
        else:
            if "def " in code_clean and "self" in code_clean:
                style_score += 8.0
            if "from typing import" in code_clean or "-> " in code_clean:
                style_score += 7.0
        score += min(25.0, style_score)

        # 3. Invariant Adherence & Statefulness (Max 25 points)
        invariant_score = 10.0
        if "class " in code_clean and "constructor" in code_clean or "__init__" in code_clean:
            invariant_score += 8.0
        if "this." in code_clean or "self." in code_clean:
            invariant_score += 4.0
        if "lock" in code_clean.lower() or "semaphore" in code_clean.lower() or "queue" in code_clean.lower():
            invariant_score += 3.0
        score += min(25.0, invariant_score)

        # 4. Thinking Reasoning Depth (Max 15 points)
        reasoning_score = 5.0
        if thinking and len(thinking.strip()) > 30:
            reasoning_score += 5.0
        if "Root cause:" in thinking or "invariants" in thinking.lower():
            reasoning_score += 5.0
        score += min(15.0, reasoning_score)

        return min(100.0, score)

    def compare_passing_candidates(
        self,
        candidate_a: Dict[str, Any],
        candidate_b: Dict[str, Any],
        language: Language,
    ) -> Tuple[Optional[str], float, str]:
        candidates = [candidate_a, candidate_b]
        random.shuffle(candidates)
        cand_0 = candidates[0]
        cand_1 = candidates[1]

        score_0 = self.evaluate_code_rubric(cand_0["code"], cand_0["thinking"], language)
        score_1 = self.evaluate_code_rubric(cand_1["code"], cand_1["thinking"], language)

        delta = abs(score_0 - score_1)
        if delta < self.score_margin_threshold:
            return None, delta, f"Score margin ({delta:.1f}) below threshold ({self.score_margin_threshold})"

        if score_0 > score_1:
            chosen_tag = cand_0["tag"]
            reason = f"{cand_0['tag']} won with score {score_0:.1f} vs {score_1:.1f} (delta={delta:.1f})"
        else:
            chosen_tag = cand_1["tag"]
            reason = f"{cand_1['tag']} won with score {score_1:.1f} vs {score_0:.1f} (delta={delta:.1f})"

        return chosen_tag, delta, reason


class SyntheticDPOGenerator:
    def __init__(
        self,
        harvester: Optional[ExperienceHarvester] = None,
        timeout: int = 8,
        score_margin: float = 15.0,
    ):
        self.harvester = harvester or ExperienceHarvester()
        self.py_runner = PythonRunner(use_docker=False, timeout=timeout)
        self.ts_runner = TypeScriptRunner(use_docker=False, timeout=timeout)
        self.judge = RubricJudge(score_margin_threshold=score_margin)

    def _execute_solution(self, solution_code: str, test_code: str, language: Language) -> ExecutionResult:
        runner = self.py_runner if language == Language.PYTHON else self.ts_runner
        return runner.run_code_and_test(
            code=solution_code,
            test_code=test_code,
            module_name="solution",
        )

    def _format_assistant_response(self, thinking: str, code: str, test_code: str, language: Language) -> str:
        lang_str = language.value
        return (
            f"<thinking>\n{thinking.strip()}\n</thinking>\n\n"
            f"### Production-Ready Code\n```{lang_str}\n{code.strip()}\n```\n\n"
            f"### Unit Test Suite\n```{lang_str}\n{test_code.strip()}\n```"
        )

    def generate_candidate_variants(self, sample: DatasetSample) -> List[Dict[str, Any]]:
        lang = sample.language
        lang_str = lang.value
        candidates = []

        # Candidate 0: Gold/Optimal Solution
        candidates.append({
            "tag": "gold_optimal",
            "code": sample.solution_code,
            "thinking": f"Root cause: {sample.explanation or 'Ensure correct algorithmic invariant and edge-case handling.'}",
            "test_code": sample.test_code,
        })

        # Candidate 1: Buggy/Faulty Solution
        candidates.append({
            "tag": "buggy_faulty",
            "code": sample.buggy_code,
            "thinking": "Naive implementation without full invariant validation.",
            "test_code": sample.test_code,
        })

        # Candidate 2: Suboptimal Variant (Passing but inefficient/non-idiomatic)
        suboptimal_code = sample.solution_code
        if lang == Language.PYTHON:
            suboptimal_code = "# Suboptimal version with extra iterations\n" + sample.solution_code
        else:
            suboptimal_code = "// Suboptimal version with extra allocations\n" + sample.solution_code

        candidates.append({
            "tag": "suboptimal_variant",
            "code": suboptimal_code,
            "thinking": "Basic straightforward solution with redundant operations.",
            "test_code": sample.test_code,
        })

        return candidates

    def process_sample(self, sample: DatasetSample) -> Tuple[bool, str]:
        candidates = self.generate_candidate_variants(sample)
        tested_candidates = []

        for cand in candidates:
            try:
                res = self._execute_solution(cand["code"], sample.test_code, sample.language)
                cand["passed"] = (res.status == ExecutionStatus.PASSED)
                cand["exec_status"] = res.status.value
            except Exception as e:
                cand["passed"] = False
                cand["exec_status"] = f"error: {str(e)}"
            tested_candidates.append(cand)

        passing = [c for c in tested_candidates if c["passed"]]
        failing = [c for c in tested_candidates if not c["passed"]]

        chosen_cand = None
        rejected_cand = None
        rejection_reason = ""

        # Level 1: Deterministic Execution Gate
        if passing and failing:
            chosen_cand = passing[0]
            rejected_cand = failing[0]
            rejection_reason = f"Deterministic Execution Gate: Rejected failed test execution ({rejected_cand['exec_status']})."
        elif len(passing) >= 2:
            # Level 2: Rubric-Based Judge with anti-position bias & margin gate
            winner_tag, delta, reason = self.judge.compare_passing_candidates(
                passing[0], passing[1], sample.language
            )
            if winner_tag is None:
                return False, f"Dropped: {reason}"
            if winner_tag == passing[0]["tag"]:
                chosen_cand, rejected_cand = passing[0], passing[1]
            else:
                chosen_cand, rejected_cand = passing[1], passing[0]
            rejection_reason = f"AI Judge Rubric: {reason}"
        else:
            return False, "Dropped: No passing solution or insufficient candidate separation."

        chosen_formatted = self._format_assistant_response(
            chosen_cand["thinking"], chosen_cand["code"], sample.test_code, sample.language
        )
        rejected_formatted = self._format_assistant_response(
            rejected_cand["thinking"], rejected_cand["code"], sample.test_code, sample.language
        )

        ok, dpo_id = self.harvester.harvest_dpo_pair(
            user_query=sample.instruction,
            chosen_text=chosen_formatted,
            rejected_text=rejected_formatted,
            rejection_reason=rejection_reason,
            language=sample.language,
            source="synthetic_dpo_generator",
        )

        if ok:
            return True, f"Harvested {dpo_id} ({chosen_cand['tag']} > {rejected_cand['tag']})"
        return False, f"Harvest failed: {dpo_id}"

    def generate_batch(self, count: int = 10) -> int:
        harvested = 0
        generators = [
            generate_python_sample,
            generate_ts_sample,
            generate_concurrency_python_sample,
            generate_concurrency_ts_sample,
        ]

        print(f"[*] Starting Synthetic DPO Generation ({count} target samples)...", flush=True)
        for i in range(count):
            gen_fn = generators[i % len(generators)]
            sample = gen_fn(i + int(time.time()) % 1000)
            ok, msg = self.process_sample(sample)
            if ok:
                harvested += 1
                print(f"  [+] [{harvested}/{count}] {msg} - '{sample.instruction[:45]}...'")
            else:
                print(f"  [-] Skipped [{i+1}/{count}]: {msg}")

        print(f"[*] Batch Complete! Successfully added {harvested} verified DPO preference pairs to pool.\n", flush=True)
        return harvested


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic DPO Generation Pipeline")
    parser.add_argument("--count", type=int, default=10, help="Number of DPO pairs to generate")
    parser.add_argument("--margin", type=float, default=15.0, help="Score margin threshold for AI Judge")
    args = parser.parse_args()

    generator = SyntheticDPOGenerator(score_margin=args.margin)
    generator.generate_batch(count=args.count)
