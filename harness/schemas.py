from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class Language(str, Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"


class ExecutionStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    SYSTEM_ERROR = "system_error"


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    error_message: Optional[str] = None


@dataclass
class VerificationOutcome:
    sample_id: str
    language: Language
    is_valid: bool
    negative_check: ExecutionResult
    positive_check: ExecutionResult
    rejection_reason: Optional[str] = None


@dataclass
class DatasetSample:
    sample_id: str
    language: Language
    category: str
    difficulty: str
    instruction: str
    buggy_code: str
    solution_code: str
    test_code: str
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
