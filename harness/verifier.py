from harness.schemas import (
    Language,
    ExecutionStatus,
    ExecutionResult,
    VerificationOutcome,
    DatasetSample,
)
from harness.python_runner import PythonRunner
from harness.ts_runner import TypeScriptRunner


class CodeVerifier:
    def __init__(self, use_docker: bool = True, timeout: int = 15):
        self.py_runner = PythonRunner(use_docker=use_docker, timeout=timeout)
        self.ts_runner = TypeScriptRunner(use_docker=use_docker, timeout=timeout)

    def verify_sample(self, sample: DatasetSample) -> VerificationOutcome:
        """
        Executes strict two-way verification on a dataset sample:
        1. Negative check: buggy_code + test_code MUST FAIL.
        2. Positive check: solution_code + test_code MUST PASS.
        """
        runner = (
            self.py_runner
            if sample.language == Language.PYTHON
            else self.ts_runner
        )

        # Step 1: Negative check (Buggy code must FAIL the test)
        neg_result = runner.run_code_and_test(
            code=sample.buggy_code,
            test_code=sample.test_code,
            module_name="solution",
        )

        if neg_result.status == ExecutionStatus.PASSED:
            return VerificationOutcome(
                sample_id=sample.sample_id,
                language=sample.language,
                is_valid=False,
                negative_check=neg_result,
                positive_check=ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    exit_code=-1,
                    stdout="",
                    stderr="Skipped due to negative check failure",
                    duration_seconds=0,
                ),
                rejection_reason="Trivial/Invalid Test: Buggy code unexpectedly passed the test harness.",
            )

        # Step 2: Positive check (Solution code must PASS the test)
        pos_result = runner.run_code_and_test(
            code=sample.solution_code,
            test_code=sample.test_code,
            module_name="solution",
        )

        if pos_result.status != ExecutionStatus.PASSED:
            return VerificationOutcome(
                sample_id=sample.sample_id,
                language=sample.language,
                is_valid=False,
                negative_check=neg_result,
                positive_check=pos_result,
                rejection_reason=f"Broken Fix: Solution code failed the test suite with exit code {pos_result.exit_code}.",
            )

        # Both passed criteria
        return VerificationOutcome(
            sample_id=sample.sample_id,
            language=sample.language,
            is_valid=True,
            negative_check=neg_result,
            positive_check=pos_result,
            rejection_reason=None,
        )
