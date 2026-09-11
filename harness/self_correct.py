import re
from dataclasses import dataclass
from typing import Optional, Tuple, List
from harness.schemas import Language, ExecutionStatus, ExecutionResult
from harness.python_runner import PythonRunner
from harness.ts_runner import TypeScriptRunner


@dataclass
class AutoVerifyResult:
    has_tests: bool
    is_valid: bool
    language: Optional[Language]
    status: str
    duration_seconds: float
    stdout: str
    stderr: str
    error_message: Optional[str] = None
    extracted_code: Optional[str] = None
    extracted_test: Optional[str] = None


def sanitize_ts_parameter_properties(code: str) -> str:
    """
    Sanitizes TypeScript parameter properties in class constructors.
    Transforms:
      constructor(private x: number, public readonly y: string = 'val') {
    into:
      private x: number;
      public readonly y: string;
      constructor(x: number, y: string = 'val') {
        this.x = x;
        this.y = y;
    """
    if not code or "constructor" not in code:
        return code

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


def sanitize_test_variable_typos(test_code: str) -> str:
    """
    Automatically sanitizes common model identifier typos in the test block:
    Finds any declared test variable instance (e.g. `let limiter` or `const limiter`)
    and normalizes misspelled variations (such as `llimiter`, `limeder`, `llims`, `limitter`, `limitr`)
    back to the exact declared variable name (`limiter`).
    """
    if not test_code:
        return test_code

    # Match declarations: let x, const x, var x, let x: Type, const x: Type
    declared_vars = re.findall(r"\b(?:let|const|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\b", test_code)
    declared_set = set(declared_vars)

    for decl in declared_vars:
        if len(decl) < 3:
            continue

        # 1. Double letter prefix: e.g. 'l' + 'limiter' -> 'llimiter', 'bbucket', 'ccounter'
        prefix_double = decl[0] + decl
        if prefix_double not in declared_set:
            test_code = re.sub(rf"\b{re.escape(prefix_double)}\b", decl, test_code)

        # 2. Specific common variations for limiter / bucket / counter
        if decl.lower() == "limiter":
            typos = ["llimiter", "limeder", "llims", "limitter", "limitr", "limt", "llim", "limtr", "limite"]
            for typo in typos:
                if typo not in declared_set:
                    test_code = re.sub(rf"\b{re.escape(typo)}\b", decl, test_code)
        elif decl.lower() == "bucket":
            typos = ["bbucket", "buckt", "buckett", "bkt"]
            for typo in typos:
                if typo not in declared_set:
                    test_code = re.sub(rf"\b{re.escape(typo)}\b", decl, test_code)
        elif decl.lower() == "counter":
            typos = ["ccounter", "countr", "counteer", "cntr"]
            for typo in typos:
                if typo not in declared_set:
                    test_code = re.sub(rf"\b{re.escape(typo)}\b", decl, test_code)

    return test_code


def sanitize_string_delimiter_mismatches(code: str) -> str:
    """
    Sanitizes mismatched string quotes in JS/TS/Python code and test blocks.
    Fixes cases like:
      it('test description", () => { -> it('test description', () => {
      assert.strictEqual(x, 'hello"); -> assert.strictEqual(x, 'hello');
      assert.strictEqual(x, "hello'); -> assert.strictEqual(x, "hello");
    """
    if not code:
        return code

    def test_call_replacer(match):
        fn = match.group(1)
        open_q = match.group(2)
        content = match.group(3)
        rest = match.group(5)
        return f"{fn}({open_q}{content}{open_q}{rest}"

    test_pattern = re.compile(
        r"\b(describe|it|test)\s*\(\s*(['\"])(.*?)(['\"])\s*(,)",
        re.DOTALL
    )
    code = test_pattern.sub(test_call_replacer, code)

    # General single-line mismatched string literals
    code = re.sub(r"'([^'\r\n\"]*?)\"([),;\s\]\}])", r"'\1'\2", code)
    code = re.sub(r"\"([^'\r\n\"]*?)'([),;\s\]\}])", r'"\1"\2', code)

    return code


def sanitize_ts_imports(test_code: str, value_names: set = None) -> str:
    """
    Sanitizes solution imports for Node.js --experimental-strip-types.
    Strips the `type` keyword from statement-level and inline import specifiers
    for runnable classes and functions (e.g. `import type { ClassName }` -> `import { ClassName }`
    and `import { type Config, RealClass }` -> `import { Config, RealClass }`).
    """
    if not test_code:
        return test_code

    # 1. Strip inline 'type ' keywords inside import { ... } from './solution'
    inline_type_pattern = re.compile(
        r"import\s*\{([^}]+)\}\s*from\s*['\"](\./solution(?:\.ts)?)['\"];?",
        re.DOTALL
    )

    def inline_replacer(match):
        inner = match.group(1)
        mod = match.group(2)
        cleaned_inner = re.sub(r"\btype\s+([a-zA-Z0-9_$]+)", r"\1", inner)
        return f"import {{ {cleaned_inner.strip()} }} from '{mod}';"

    test_code = inline_type_pattern.sub(inline_replacer, test_code)

    # 2. Normalize statement-level `import type { ... } from './solution'`
    stmt_type_pattern = re.compile(
        r"import\s+type\s*\{([^}]+)\}\s*from\s*['\"](\./solution(?:\.ts)?)['\"];?",
        re.DOTALL
    )

    def stmt_replacer(match):
        inner = match.group(1).strip()
        mod = match.group(2)
        symbols = [s.strip() for s in inner.split(",") if s.strip()]

        if value_names:
            vals = [s for s in symbols if s in value_names]
            types = [s for s in symbols if s not in value_names]
            res = []
            if vals:
                res.append(f"import {{ {', '.join(vals)} }} from '{mod}';")
            if types:
                res.append(f"import type {{ {', '.join(types)} }} from '{mod}';")
            return "\n".join(res)
        else:
            return f"import {{ {inner} }} from '{mod}';"

    test_code = stmt_type_pattern.sub(stmt_replacer, test_code)
    return test_code


def break_degenerate_repetition(text: str) -> str:
    """
    Detects and breaks degenerate token loops (e.g., repeating sequences like 'TS TS TS...',
    repeated statements, or looping tokens) to prevent infinite or repetitive collapses.
    """
    if not text or len(text) < 20:
        return text

    # 1. Regex to catch the exact SAME word/token repeated 4+ times: e.g. "TS TS TS TS..."
    text = re.sub(r"\b([a-zA-Z0-9_$#@.-]{2,30})\b(?:\s+\1\b){3,}", r"\1", text)

    # 2. Multi-line repetition: truncate identical consecutive non-trivial lines repeated 3+ times
    # (Ignore single closing braces or simple punctuation like '}', '};', '});', ']', ')')
    ignore_lines = {"}", "{", "};", "});", "]", ")", ";", ""}
    lines = text.splitlines()
    cleaned_lines = []
    repeat_count = 0
    last_line = None

    for line in lines:
        stripped = line.strip()
        if stripped and stripped not in ignore_lines and stripped == last_line:
            repeat_count += 1
            if repeat_count < 2:
                cleaned_lines.append(line)
        else:
            repeat_count = 0
            last_line = stripped if stripped and stripped not in ignore_lines else None
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _detect_language(lang_tag: str, content: str) -> Optional[Language]:
    tag = lang_tag.strip().lower()
    if tag in ["typescript", "ts", "tsx", "javascript", "js", "jsx"] or tag.startswith("ts") or tag.startswith("js"):
        return Language.TYPESCRIPT
    if tag in ["python", "py", "python3"] or tag.startswith("py"):
        return Language.PYTHON
    if "export " in content or ("import " in content and ("from './" in content or "from \"./" in content or "node:" in content)):
        return Language.TYPESCRIPT
    if "def " in content or ("import " in content and ("pytest" in content or "solution" in content)):
        return Language.PYTHON
    return None


def _split_combined_block(content: str, lang: Language) -> Tuple[Optional[str], Optional[str]]:
    content_strip = content.strip()
    if lang == Language.PYTHON:
        match = re.search(r"\n(?=(?:@pytest\.|def\s+test_))", content_strip)
        if match and match.start() > 0:
            impl = content_strip[:match.start()].strip()
            test = content_strip[match.start():].strip()
            has_impl = bool(re.search(r"^\s*(?:def\s+(?!test_)|class\s+)", impl, re.MULTILINE))
            if impl and test and has_impl:
                return impl, test
    else:
        match = re.search(r"\n(?=(?:describe\s*\(|it\s*\(|test\s*\())", content_strip)
        if match and match.start() > 0:
            impl = content_strip[:match.start()].strip()
            test = content_strip[match.start():].strip()
            has_impl = bool(re.search(r"^\s*(?:export\s+)?(?:class|function|interface|enum)\s+[a-zA-Z0-9_$]+", impl, re.MULTILINE))
            has_test = bool(re.search(r"\b(?:describe|it|test)\s*\(", test))
            if impl and test and has_impl and has_test:
                return impl, test
    return None, None


def extract_code_and_tests(text: str) -> Tuple[Optional[Language], Optional[str], Optional[str]]:
    """
    Parses generated markdown response to isolate implementation code and unit tests.
    Supports Windows/Unix newlines, heading-based partitioning, and in-block test decoupling.
    """
    if not text:
        return None, None, None

    text = break_degenerate_repetition(text)
    code_block_pattern = re.compile(r"```([a-zA-Z0-9_-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
    matches = code_block_pattern.findall(text)

    if not matches:
        return None, None, None

    impl_blocks: List[str] = []
    test_blocks: List[str] = []
    detected_lang: Optional[Language] = None

    # Check if headings exist in text to partition matches
    heading_sections = re.split(r"(?i)###\s*(?:Unit\s*Test\s*Suite|Tests?|Test\s*Suite|Test\s*Cases?)", text, maxsplit=1)
    if len(heading_sections) == 2:
        prod_part, test_part = heading_sections
        prod_matches = code_block_pattern.findall(prod_part)
        test_matches = code_block_pattern.findall(test_part)

        for lang_tag, content in prod_matches:
            if content.strip():
                impl_blocks.append(content.strip())
                if not detected_lang:
                    detected_lang = _detect_language(lang_tag, content)

        for lang_tag, content in test_matches:
            if content.strip():
                test_blocks.append(content.strip())
                if not detected_lang:
                    detected_lang = _detect_language(lang_tag, content)

    if not impl_blocks and not test_blocks:
        for idx, (lang_tag, content) in enumerate(matches):
            content_strip = content.strip()
            if not detected_lang:
                detected_lang = _detect_language(lang_tag, content_strip)
            if not detected_lang:
                detected_lang = Language.PYTHON

            # Check if this single block contains both implementation and tests
            impl_part, test_part = _split_combined_block(content_strip, detected_lang)
            if impl_part and test_part:
                impl_blocks.append(impl_part)
                test_blocks.append(test_part)
                continue

            is_test = False
            if detected_lang == Language.PYTHON:
                if ("def test_" in content_strip or "import pytest" in content_strip or 
                    "from solution import" in content_strip or "pytest." in content_strip or
                    ("assert " in content_strip and "def test" in content_strip)):
                    is_test = True
            else:
                if ("node:test" in content_strip or "describe(" in content_strip or 
                    "it(" in content_strip or "from './solution" in content_strip or 
                    "expect(" in content_strip or "test(" in content_strip):
                    is_test = True

            if not is_test and len(matches) == 2 and idx == 1:
                if "assert" in content_strip or "test" in content_strip or "expect" in content_strip:
                    is_test = True

            if is_test:
                test_blocks.append(content_strip)
            else:
                impl_blocks.append(content_strip)

    if not detected_lang:
        detected_lang = Language.PYTHON

    # Decouple any remaining unified blocks in impl_blocks
    cleaned_impl_blocks: List[str] = []
    for block in impl_blocks:
        impl_part, test_part = _split_combined_block(block, detected_lang)
        if impl_part and test_part:
            cleaned_impl_blocks.append(impl_part)
            test_blocks.append(test_part)
        else:
            cleaned_impl_blocks.append(block)

    impl_code = "\n\n".join(b for b in cleaned_impl_blocks if b) if cleaned_impl_blocks else None
    test_code = "\n\n".join(b for b in test_blocks if b) if test_blocks else None

    # Auto-sanitize string delimiter mismatches across all languages
    if impl_code:
        impl_code = sanitize_string_delimiter_mismatches(impl_code)
    if test_code:
        test_code = sanitize_string_delimiter_mismatches(test_code)

    # Auto-add or normalize imports and sanitize TypeScript syntax
    if impl_code and test_code:
        if detected_lang == Language.PYTHON:
            if "from solution import" not in test_code and "import solution" not in test_code:
                fn_names = re.findall(r"^def\s+([a-zA-Z0-9_]+)\s*\(", impl_code, re.MULTILINE)
                class_names = re.findall(r"^class\s+([a-zA-Z0-9_]+)", impl_code, re.MULTILINE)
                names_to_import = [n for n in (fn_names + class_names) if not n.startswith("_")]
                if names_to_import:
                    import_stmt = f"from solution import {', '.join(names_to_import)}\n"
                    test_code = import_stmt + test_code
        else:
            # 1. Sanitize constructor parameter properties for Node.js strip-only mode
            impl_code = sanitize_ts_parameter_properties(impl_code)
            test_code = sanitize_ts_parameter_properties(test_code)

            # 2. Sanitize common test variable identifier typos
            test_code = sanitize_test_variable_typos(test_code)

            # 3. Detect TypeScript interfaces/types vs runtime values
            type_names = set(re.findall(r"export\s+(?:interface|type)\s+([a-zA-Z0-9_]+)", impl_code))
            value_names = set(re.findall(r"export\s+(?:class|function|const|let|enum)\s+([a-zA-Z0-9_]+)", impl_code))

            # 4. Sanitize statement-level and inline 'type' imports for runtime values
            test_code = sanitize_ts_imports(test_code, value_names)

            # 5. Normalize any existing imports in test_code where interfaces were erroneously imported as values
            for t_name in type_names:
                val_import_pattern = rf"import\s*\{{([^}}]*\b{t_name}\b[^}}]*)\}}\s*from\s*['\"](\./solution(?:\.ts)?)['\"];?"
                match = re.search(val_import_pattern, test_code)
                if match and "import type" not in match.group(0):
                    inner_symbols = [s.strip() for s in match.group(1).split(",") if s.strip()]
                    remaining_values = [s for s in inner_symbols if s != t_name and s not in type_names]
                    new_imports = []
                    if remaining_values:
                        new_imports.append(f"import {{ {', '.join(remaining_values)} }} from '{match.group(2)}';")
                    new_imports.append(f"import type {{ {t_name} }} from '{match.group(2)}';")
                    test_code = test_code[:match.start()] + "\n".join(new_imports) + test_code[match.end():]

            # 6. If no solution import exists at all, auto-generate cleanly separated imports
            if "from './solution" not in test_code and 'from "./solution' not in test_code:
                import_stmts = []
                if value_names:
                    import_stmts.append(f"import {{ {', '.join(sorted(value_names))} }} from './solution.ts';")
                if type_names:
                    import_stmts.append(f"import type {{ {', '.join(sorted(type_names))} }} from './solution.ts';")
                if import_stmts:
                    test_code = "\n".join(import_stmts) + "\n" + test_code

            # 7. Node.js 'expect' compatibility shim injection
            if "expect(" in test_code and "const expect" not in test_code and "function expect" not in test_code:
                expect_shim = """import assert from 'node:assert/strict';
import { describe, it, test, beforeEach, afterEach, before, after } from 'node:test';

const expect = (val: any) => ({
  toBe: (exp: any) => assert.strictEqual(val, exp),
  toEqual: (exp: any) => assert.deepStrictEqual(val, exp),
  toStrictEqual: (exp: any) => assert.deepStrictEqual(val, exp),
  toBeUndefined: () => assert.strictEqual(val, undefined),
  toBeNull: () => assert.strictEqual(val, null),
  toBeDefined: () => assert.notStrictEqual(val, undefined),
  toBeTruthy: () => assert.ok(val),
  toBeFalsy: () => assert.ok(!val),
  toHaveLength: (len: number) => assert.strictEqual(val?.length, len),
  toContain: (item: any) => assert.ok(val && val.includes ? val.includes(item) : false),
  toThrow: (expected?: any) => {
    if (typeof val === 'function') {
      assert.throws(val, expected);
    } else {
      assert.throws(() => val, expected);
    }
  },
  rejects: {
    toThrow: async (expected?: any) => await assert.rejects(val, expected),
    toBe: async (exp: any) => assert.strictEqual(await val, exp),
    toEqual: async (exp: any) => assert.deepStrictEqual(await val, exp),
  }
});
"""
                test_code = expect_shim + "\n" + test_code

            # 8. Normalize node:test imports to include all lifecycle hooks
            node_test_import = "import { describe, it, test, beforeEach, afterEach, before, after } from 'node:test';"
            if "node:test" in test_code:
                test_code = re.sub(r"import\s*\{[^}]*\}\s*from\s*['\"]node:test['\"];?", node_test_import, test_code)
            elif any(fn in test_code for fn in ["describe(", "it(", "test(", "beforeEach(", "afterEach(", "before(", "after("]):
                test_code = node_test_import + "\n" + test_code

            # 9. Normalize node:assert imports
            if "assert." in test_code and "node:assert" not in test_code:
                test_code = "import assert from 'node:assert/strict';\n" + test_code

    return detected_lang, impl_code, test_code


class SelfCorrectionEngine:
    def __init__(self, timeout: int = 6):
        self.py_runner = PythonRunner(use_docker=False, timeout=timeout)
        self.ts_runner = TypeScriptRunner(use_docker=False, timeout=timeout)

    def verify_output(self, text: str) -> AutoVerifyResult:
        lang, impl_code, test_code = extract_code_and_tests(text)

        if not impl_code or not test_code:
            return AutoVerifyResult(
                has_tests=False,
                is_valid=True,
                language=lang,
                status="no_tests_found",
                duration_seconds=0.0,
                stdout="",
                stderr="",
                error_message=None,
                extracted_code=impl_code,
                extracted_test=test_code,
            )

        runner = self.py_runner if lang == Language.PYTHON else self.ts_runner
        result: ExecutionResult = runner.run_code_and_test(
            code=impl_code,
            test_code=test_code,
            module_name="solution",
        )

        is_passed = (result.status == ExecutionStatus.PASSED)

        return AutoVerifyResult(
            has_tests=True,
            is_valid=is_passed,
            language=lang,
            status=result.status.value,
            duration_seconds=result.duration_seconds,
            stdout=result.stdout,
            stderr=result.stderr,
            error_message=result.error_message if not is_passed else None,
            extracted_code=impl_code,
            extracted_test=test_code,
        )

    def build_reflection_prompt(self, user_query: str, last_response: str, verify_res: AutoVerifyResult) -> str:
        error_info = verify_res.error_message or verify_res.stderr or verify_res.stdout
        error_snippet = "\n".join(error_info.strip().splitlines()[-14:])
        error_lower = error_info.lower()

        test_source = (verify_res.extracted_test or "") + " " + last_response
        has_unparam_calls = bool(re.search(r"allow(?:Request)?\(\s*\)", test_source))
        has_consecutive_zero_time = has_unparam_calls or bool(re.search(r"allow(?:Request)?\(\s*t0\s*\).+?allow(?:Request)?\(\s*t0\s*\)", test_source, re.DOTALL))

        # Check if user query or implementation is specifically about rate-limiting, timing, or concurrency
        is_timing_or_limiter = any(kw in (user_query + " " + test_source).lower() for kw in ["rate", "limiter", "bucket", "sliding", "window", "ttl", "timestamp", "debounce", "interval", "mutex", "lock", "concurren"])

        diagnostics_flags = []
        if "referenceerror" in error_lower or "nameerror" in error_lower or "is not defined" in error_lower:
            diagnostics_flags.append("- ReferenceError / NameError: A variable or identifier name is undefined or misspelled. Fix the variable name to match the argument list exactly.")

        if "syntaxerror" in error_lower or "err_unsupported_typescript_syntax" in error_lower:
            diagnostics_flags.append("- SyntaxError: Fix string quote delimiter mismatches (e.g. `it('...', ...)`), constructor parameter properties, or malformed braces. Do not alter core problem logic.")

        if "assertionerror" in error_lower or "err_assertion" in error_lower or "assert" in error_lower:
            if has_consecutive_zero_time and is_timing_or_limiter:
                diagnostics_flags.append("- Zero Elapsed Time Warning: Assertion failed due to zero elapsed time. Do not change working algorithm logic. Fix the unit test by passing explicit advancing timestamps (e.g. now + windowMs + 1).")
            if is_timing_or_limiter:
                diagnostics_flags.append("- State Mutation & Timing: Verify that state updates (e.g. array filtering, counters, timestamps) are properly assigned to instance fields and tested with monotonic mock timestamps.")

        diag_section = ""
        if diagnostics_flags:
            diag_section = "\n### Auto-Diagnostics Flags\n" + "\n".join(diagnostics_flags) + "\n"

        prompt = f"""Your previous code for the requested task failed execution or unit test assertions.

### Original User Task:
"{user_query}"

### Error Diagnostics:
```
{error_snippet}
```{diag_section}
### Strict Self-Correction Directives:
1. Target Task Isolation: Fix ONLY the specific syntax, type, or assertion errors in your previous solution for the exact task: "{user_query}".
2. STRICT ANTI-DRIFT CONSTRAINT: Do NOT change the problem statement. Do NOT invent unrelated classes (e.g., do NOT generate rate limiters, token buckets, or complex caches unless specifically requested by the user in "{user_query}").
3. Syntax & Quote Integrity: Ensure all string delimiters in tests and implementation are correctly matched (e.g. `'...'` or `"..."`). Never mix single and double quotes on the same string literal.
4. TypeScript & ESM Rules: In TypeScript unit tests, use standard value imports (`import {{ ... }} from './solution.ts'`) and native `node:test` / `node:assert/strict`.
5. Format Requirement: Output the revised `<thinking>` analysis, corrected code block under `### Production-Ready Code`, and passing tests under `### Unit Test Suite`."""
        return prompt
        return prompt
