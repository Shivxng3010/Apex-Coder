from typing import Dict, Any
from harness.schemas import DatasetSample, Language


SYSTEM_ASSISTANT_PROMPT = (
    "You are a precise, senior software engineering assistant specializing in "
    "diagnosing bugs, implementing robust fixes, and writing comprehensive test suites."
)


def format_sample_to_chatml(sample: DatasetSample) -> Dict[str, Any]:
    """
    Formats a verified dataset sample into standard OpenAI / Qwen ChatML message schema
    ready for QLoRA SFT training.
    """
    lang_name = "python" if sample.language == Language.PYTHON else "typescript"

    user_content = f"""Please debug the following {sample.language.value.capitalize()} code.

### Problem Description
{sample.instruction}

### Buggy Code
```{lang_name}
{sample.buggy_code.strip()}
```

Identify the bug, provide the corrected implementation, and supply a verified unit test suite."""

    assistant_content = f"""### Analysis & Root Cause
{sample.explanation or "The implementation contained logic/typing issues that resulted in unexpected behavior."}

### Corrected Implementation
```{lang_name}
{sample.solution_code.strip()}
```

### Unit Test Suite
```{lang_name}
{sample.test_code.strip()}
```"""

    return {
        "id": sample.sample_id,
        "language": sample.language.value,
        "category": sample.category,
        "difficulty": sample.difficulty,
        "messages": [
            {"role": "system", "content": SYSTEM_ASSISTANT_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }
