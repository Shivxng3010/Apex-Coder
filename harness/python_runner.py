import os
import sys
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional
from harness.schemas import ExecutionResult, ExecutionStatus, Language


def _kill_process_tree(proc: subprocess.Popen):
    """Safely kills a process and all its subprocesses across platforms."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            proc.kill()
    except Exception:
        pass


class PythonRunner:
    def __init__(
        self,
        use_docker: bool = False,
        docker_image: str = "coder-verifier-py:latest",
        timeout: int = 5,
    ):
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.timeout = timeout

    def run_code_and_test(
        self,
        code: str,
        test_code: str,
        module_name: str = "solution",
        run_mypy: bool = False,
    ) -> ExecutionResult:
        """
        Executes Python code and tests inside a sandboxed environment.
        1. Writes code to <module_name>.py
        2. Writes test_code to test_<module_name>.py
        3. Runs pytest with timeout
        """
        start_time = time.time()
        
        with tempfile.TemporaryDirectory(prefix="verifier_py_") as temp_dir:
            temp_path = Path(temp_dir)
            code_file = temp_path / f"{module_name}.py"
            test_file = temp_path / f"test_{module_name}.py"

            code_file.write_text(code, encoding="utf-8")
            test_file.write_text(test_code, encoding="utf-8")

            if self.use_docker and self._is_docker_available():
                return self._run_in_docker(temp_path, module_name, start_time)
            else:
                return self._run_local(temp_path, module_name, start_time)

    def _is_docker_available(self) -> bool:
        try:
            res = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2,
            )
            return res.returncode == 0
        except Exception:
            return False

    def _run_in_docker(
        self, temp_path: Path, module_name: str, start_time: float
    ) -> ExecutionResult:
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--network", "none",
            "--memory", "512m",
            "--cpus", "1.0",
            "-v", f"{str(temp_path.resolve())}:/sandbox:ro",
            self.docker_image,
            "pytest",
            f"test_{module_name}.py",
            "-q",
            "--tb=short",
        ]

        proc = subprocess.Popen(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            stdout, stderr = proc.communicate(timeout=self.timeout + 3)
            duration = time.time() - start_time
            if proc.returncode == 0:
                return ExecutionResult(
                    status=ExecutionStatus.PASSED,
                    exit_code=0,
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=duration,
                )
            else:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    exit_code=proc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=duration,
                    error_message=stderr or stdout,
                )
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                exit_code=-1,
                stdout="",
                stderr="Execution timed out.",
                duration_seconds=time.time() - start_time,
                error_message="Test execution timed out.",
            )
        except Exception as e:
            _kill_process_tree(proc)
            return ExecutionResult(
                status=ExecutionStatus.SYSTEM_ERROR,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration_seconds=time.time() - start_time,
                error_message=str(e),
            )

    def _run_local(
        self, temp_path: Path, module_name: str, start_time: float
    ) -> ExecutionResult:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(temp_path / f"test_{module_name}.py"),
            "-q",
            "--tb=short",
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(temp_path.resolve())

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(temp_path),
            env=env,
        )

        try:
            stdout, stderr = proc.communicate(timeout=self.timeout)
            duration = time.time() - start_time
            if proc.returncode == 0:
                return ExecutionResult(
                    status=ExecutionStatus.PASSED,
                    exit_code=0,
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=duration,
                )
            else:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    exit_code=proc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=duration,
                    error_message=stderr or stdout,
                )
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                exit_code=-1,
                stdout="",
                stderr="Execution timed out.",
                duration_seconds=time.time() - start_time,
                error_message="Test execution timed out.",
            )
        except Exception as e:
            _kill_process_tree(proc)
            return ExecutionResult(
                status=ExecutionStatus.SYSTEM_ERROR,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration_seconds=time.time() - start_time,
                error_message=str(e),
            )
