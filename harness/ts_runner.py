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


def _sanitize_parameter_properties(code: str) -> str:
    """Sanitizes constructor parameter properties for Node.js strip-only mode."""
    if not code or "constructor" not in code:
        return code

    import re
    constructor_pattern = re.compile(
        r"(constructor\s*\(([^)]*)\)\s*\{)",
        re.MULTILINE
    )

    def constructor_replacer(match):
        full_ctor_decl = match.group(1)
        param_str = match.group(2)

        modifier_keywords = ["private", "public", "protected", "readonly"]
        has_param_props = any(kw in param_str for kw in modifier_keywords)
        if not has_param_props:
            return full_ctor_decl

        params = []
        current_param = []
        depth = 0
        for char in param_str:
            if char in "([{<":
                depth += 1
            elif char in ")]}>":
                depth -= 1
            elif char == "," and depth == 0:
                params.append("".join(current_param).strip())
                current_param = []
                continue
            current_param.append(char)
        if current_param:
            params.append("".join(current_param).strip())

        new_params = []
        field_declarations = []
        field_assignments = []

        param_prop_regex = re.compile(
            r"^((?:(?:private|public|protected|readonly)\s+)+)([a-zA-Z0-9_$]+)(\??)(?:\s*:\s*([^=]+))?(?:\s*=\s*(.+))?$"
        )

        for p in params:
            m = param_prop_regex.match(p)
            if m:
                mods = m.group(1).strip()
                name = m.group(2)
                optional = m.group(3) or ""
                type_ann = m.group(4).strip() if m.group(4) else "any"
                default_val = f" = {m.group(5).strip()}" if m.group(5) else ""

                decl = f"{mods} {name}{optional}: {type_ann};"
                field_declarations.append(decl)
                field_assignments.append(f"    this.{name} = {name};")

                cleaned_param = f"{name}{optional}: {type_ann}{default_val}"
                new_params.append(cleaned_param)
            else:
                new_params.append(p)

        if not field_declarations:
            return full_ctor_decl

        cleaned_param_str = ", ".join(new_params)
        declarations_block = "\n  ".join(field_declarations)
        assignments_block = "\n".join(field_assignments)

        return f"{declarations_block}\n  constructor({cleaned_param_str}) {{\n{assignments_block}"

    return constructor_pattern.sub(constructor_replacer, code)


class TypeScriptRunner:
    def __init__(
        self,
        use_docker: bool = False,
        docker_image: str = "coder-verifier-ts:latest",
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
        run_tsc: bool = False,
    ) -> ExecutionResult:
        """
        Executes TypeScript code and tests inside a sandboxed environment.
        1. Sanitizes constructor parameter properties for Node.js strip-only mode.
        2. Writes code to <module_name>.ts
        3. Writes test_code to <module_name>.test.ts
        4. Executes node native test runner with timeout
        """
        start_time = time.time()
        clean_code = _sanitize_parameter_properties(code)
        clean_test = _sanitize_parameter_properties(test_code)

        with tempfile.TemporaryDirectory(prefix="verifier_ts_") as temp_dir:
            temp_path = Path(temp_dir)
            code_file = temp_path / f"{module_name}.ts"
            test_file = temp_path / f"{module_name}.test.ts"

            code_file.write_text(clean_code, encoding="utf-8")
            test_file.write_text(clean_test, encoding="utf-8")

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
            "node",
            "--test",
            f"{module_name}.test.ts",
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
                error_message="Test timed out.",
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
        cmd = ["node", "--test", str(temp_path / f"{module_name}.test.ts")]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(temp_path),
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
                error_message="Test timed out.",
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
