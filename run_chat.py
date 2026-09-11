import os
import sys
import time
import torch
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

import gc
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness.self_correct import SelfCorrectionEngine, AutoVerifyResult, break_degenerate_repetition
from harness.experience_harvester import ExperienceHarvester
from training.auto_learn import get_active_adapter_path, execute_continual_learning
from pipeline.problem_bank import generate_python_sample, generate_ts_sample
from harness.verifier import CodeVerifier
from pipeline.synthetic_dpo_generator import SyntheticDPOGenerator
from harness.context_manager import ContextBudgetManager, strip_thinking_tags


# Configure environment paths
if Path("E:/hf_cache").exists():
    os.environ.setdefault("HF_HOME", "E:/hf_cache")
    os.environ.setdefault("UV_CACHE_DIR", "E:/uv_cache")
    os.environ.setdefault("TORCH_HOME", "E:/hf_cache/torch")


SYSTEM_PROMPT = (
    "You are Apex Coder, an expert software engineer and systems assistant.\n"
    "Before writing any solution, always think step-by-step inside <thinking> tags:\n"
    "1. Exact Task Adherence: Solve the exact problem asked by the user without inventing unrelated algorithms or overcomplicating simple tasks.\n"
    "2. Architecture & Statefulness: When building stateful utilities (queues, caches, rate limiters, mutexes), implement them as clean stateful classes.\n"
    "3. Constructor Invariant: In classes, explicitly declare all properties as class fields outside constructor and assign `this.x = x` inside. Never use parameter properties (e.g. constructor(private x: Type)).\n"
    "4. Delimiter & Syntax Integrity: Always match string quotes correctly (e.g. `it('text', ...)` or `it(\"text\", ...)`). Never mix single and double quotes on the same string literal.\n"
    "5. Testing Mandate:\n"
    "   - In TypeScript, always import runnable solution elements via standard value imports (`import { ... } from './solution.ts';`) and use native `import { describe, it } from 'node:test';` and `import assert from 'node:assert/strict';`.\n"
    "   - In Python, use `import pytest` and standard assertion statements.\n"
    "6. Timing & Virtual Time: For time-dependent utilities (rate limiters, token buckets, caches), accept optional current timestamp parameters in methods and use monotonic mock timestamps in unit tests.\n"
    "7. Output Format: You MUST format your final response with exactly two distinct markdown sections separated by headings:\n"
    "### Production-Ready Code\n"
    "```{language}\n"
    "// Complete implementation only\n"
    "```\n\n"
    "### Unit Test Suite\n"
    "```{language}\n"
    "// Standalone unit tests only\n"
    "```"
)


def resolve_model_path() -> str:
    unsloth_bnb_dir = Path("E:/hf_cache/hub/models--unsloth--Qwen2.5-Coder-3B-Instruct-bnb-4bit/snapshots/84f4805873850233fe5b2b156d3e3435425b4098")
    if unsloth_bnb_dir.exists():
        return str(unsloth_bnb_dir.resolve())
    return "Qwen/Qwen2.5-Coder-3B-Instruct"


def unload_chat_model(model=None, tokenizer=None):
    """
    Safely deallocates inference model and tokenizer from VRAM before training.
    """
    vram_before = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    vram_after = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    return max(0.0, vram_before - vram_after)


def load_chat_model(lora_adapter_dir: str = None):
    if lora_adapter_dir is None:
        lora_adapter_dir = str(get_active_adapter_path())
    base_model_path = resolve_model_path()
    adapter_path = Path(lora_adapter_dir)
    root = Path(__file__).resolve().parent
    if not adapter_path.exists():
        for cand in ["apex_coder_3b_lora_v5", "apex_coder_3b_lora_v4", "apex_coder_3b_lora_v3", "apex_coder_3b_lora_v2", "apex_coder_3b_lora"]:
            cand_p = root / "outputs" / cand
            if cand_p.exists():
                adapter_path = cand_p
                break

    print("===========================================================")
    print("        APEX CODER - INTERACTIVE CLI CHAT ENGINE")
    print("===========================================================")
    print(f"Base Model: {base_model_path}")
    print(f"LoRA Adapter: {adapter_path.resolve()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("-----------------------------------------------------------\n", flush=True)

    print("[1/2] Loading 4-bit base model into GPU memory...", end=" ", flush=True)
    t0 = time.time()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = {"": 0} if torch.cuda.is_available() else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=device_map,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    print(f"Done ({time.time()-t0:.2f}s)", flush=True)

    print("[2/2] Attaching fine-tuned LoRA adapter weights...", end=" ", flush=True)
    t1 = time.time()
    if (adapter_path / "adapter_model.safetensors").exists():
        model = PeftModel.from_pretrained(model, str(adapter_path))
    else:
        checkpoints = sorted(
            [d for d in adapter_path.glob("checkpoint-*") if (d / "adapter_model.safetensors").exists()],
            key=lambda p: int(p.name.split("-")[-1])
        )
        if checkpoints:
            latest = checkpoints[-1]
            print(f"(using {latest.name})", end=" ")
            model = PeftModel.from_pretrained(model, str(latest))
    print(f"Done ({time.time()-t1:.2f}s)", flush=True)

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None

    vram = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    print(f"\n[Ready] Model online | VRAM allocated: {vram:.2f} GB / 4.00 GB", flush=True)
    print("Engine: CoT Reasoning + Auto-Sandbox Verification Harness Active")
    print("Commands: /stats (view learning status), /learn (micro-train), /selfplay [N], /clear, /exit\n")
    print("=" * 60 + "\n", flush=True)

    return model, tokenizer


def generate_response(model, tokenizer, messages, stream: bool = True, max_new_tokens: int = 1024) -> str:
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=0.15,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return break_degenerate_repetition(raw_text)


def handle_generation_and_verification(
    model,
    tokenizer,
    messages,
    verifier: SelfCorrectionEngine,
    harvester: ExperienceHarvester,
    user_query: str,
) -> str:
    # Pass 1 Generation
    print("Apex Coder: ", end="", flush=True)
    response_p1 = generate_response(model, tokenizer, messages, stream=True)

    # Sandbox Verification Hook
    verify_res: AutoVerifyResult = verifier.verify_output(response_p1)

    if not verify_res.has_tests:
        print("\n\n[INFO] Implementation output generated (no unit test suite attached).", flush=True)
        return response_p1

    if verify_res.is_valid:
        lang_label = verify_res.language.value.capitalize() if verify_res.language else "Code"
        print(f"\n\n[PASS: VERIFIED] Sandbox Harness: 100% Passed ({lang_label} test suite passed in {verify_res.duration_seconds:.2f}s)", flush=True)
        ok, sample_id = harvester.harvest(user_query, response_p1, verify_res, source="chat_pass1")
        if ok:
            print(f"[EXPERIENCE HARVESTED] Logged verified sample ({sample_id}) into live training pool.", flush=True)
        return response_p1

    # Pass 2: Auto-Reflection & Correction Loop
    print(f"\n\n[WARN: TEST REGRESSION DETECTED] ({verify_res.error_message or 'Assertion failed'})")
    print("[*] Initiating Pass 2: Self-Correction Loop with Sandbox Diagnostics...", flush=True)

    reflection_prompt = verifier.build_reflection_prompt(user_query, response_p1, verify_res)
    refined_messages = list(messages)
    refined_messages.append({"role": "assistant", "content": response_p1})
    refined_messages.append({"role": "user", "content": reflection_prompt})

    print("\nApex Coder (Corrected): ", end="", flush=True)
    response_p2 = generate_response(model, tokenizer, refined_messages, stream=True)

    # Verify Pass 2 output
    verify_res_p2 = verifier.verify_output(response_p2)
    if verify_res_p2.is_valid:
        print(f"\n\n[PASS: AUTO-REPAIRED] Pass 2 Self-Correction: 100% Passed ({verify_res_p2.duration_seconds:.2f}s)", flush=True)
        ok, sample_id = harvester.harvest(user_query, response_p2, verify_res_p2, source="chat_pass2")
        if ok:
            print(f"[EXPERIENCE HARVESTED] Logged repaired sample ({sample_id}) into live training pool.", flush=True)

        # Harvest DPO Negative Preference Pair (chosen = Pass 2 repaired, rejected = Pass 1 buggy)
        dpo_ok, dpo_id = harvester.harvest_dpo_pair(
            user_query=user_query,
            chosen_text=response_p2,
            rejected_text=response_p1,
            rejection_reason=verify_res.error_message or "Initial generation failed unit test assertion",
            language=verify_res_p2.language,
            source="chat_self_correction"
        )
        if dpo_ok:
            print(f"[DPO PREFERENCE LOGGED] Captured chosen vs rejected alignment pair ({dpo_id}).", flush=True)
    else:
        print(f"\n\n[WARN: REVIEW RECOMMENDED] Pass 2 status: {verify_res_p2.status}", flush=True)

    return response_p2


def chat_loop(model, tokenizer, prompt_once: str = None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    verifier = SelfCorrectionEngine(timeout=6)
    harvester = ExperienceHarvester()
    budget_mgr = ContextBudgetManager(max_context_tokens=32768, budget_cap_tokens=16384, max_history_turns=16)

    if prompt_once:
        messages.append({"role": "user", "content": prompt_once})
        handle_generation_and_verification(model, tokenizer, messages, verifier, harvester, prompt_once)
        return

    while True:
        try:
            # Check if pending experience pool reached threshold (e.g. 50 samples)
            stats = harvester.get_stats()
            if stats["pending_gold_samples"] >= 50:
                print(f"\n[*] [AUTONOMOUS NOTICE] {stats['pending_gold_samples']} verified samples accumulated in experience pool.")
                print("    Type '/learn' to trigger VRAM-safe micro-training now, or continue chatting.")

            # Display Context Status HUD
            hud_line = budget_mgr.get_hud_status(messages, tokenizer)
            print(f"\n{hud_line}")

            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["/exit", "exit", "quit", ":q"]:
                print("Goodbye!")
                break

            if user_input.lower() in ["/clear", "clear", "cls"]:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                budget_mgr.rolling_memory.clear()
                print("\n[*] Conversation history and rolling memory cleared.\n")
                continue

            if user_input.lower() in ["/stats", "stats"]:
                stats = harvester.get_stats()
                active = get_active_adapter_path()
                vram = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                cur_tokens = budget_mgr.count_tokens(messages, tokenizer)
                print("\n===========================================================")
                print("               APEX CODER SYSTEM STATS")
                print("===========================================================")
                print(f"Active LoRA Adapter : {active.name} ({active.resolve()})")
                print(f"VRAM Allocated      : {vram:.2f} GB / 4.00 GB")
                print(f"Active Context Tokens: {cur_tokens:,} / 32,768 (Budget Cap: 16,384)")
                print(f"History Turns       : {(len(messages)-1)//2} turns | Memory Summaries: {len(budget_mgr.rolling_memory)}")
                print(f"Pending Gold Samples: {stats['pending_gold_samples']}")
                print(f"Pending DPO Pairs   : {stats.get('pending_dpo_pairs', 0)}")
                print(f"Consumed History    : {stats['consumed_history_samples']}")
                print(f"Experience Pool Dir : {stats['pool_dir']}")
                print("===========================================================\n")
                continue

            if user_input.lower().startswith("/selfplay"):
                count = 3
                parts = user_input.split()
                if len(parts) > 1 and parts[1].isdigit():
                    count = min(20, int(parts[1]))
                print(f"\n[*] Running CPU-based Self-Play Challenge Generator ({count} challenges)...")
                verifier_code = CodeVerifier(use_docker=False, timeout=8)
                added = 0
                for i in range(count):
                    sample = generate_python_sample(1000 + i) if i % 2 == 0 else generate_ts_sample(1000 + i)
                    outcome = verifier_code.verify_sample(sample)
                    if outcome.is_valid:
                        res_dummy = AutoVerifyResult(
                            has_tests=True,
                            is_valid=True,
                            language=sample.language,
                            status="passed",
                            duration_seconds=outcome.positive_check.duration_seconds,
                            stdout=outcome.positive_check.stdout,
                            stderr=outcome.positive_check.stderr,
                            extracted_code=sample.solution_code,
                            extracted_test=sample.test_code
                        )
                        formatted_assistant = (
                            f"<thinking>\nRoot cause: {sample.explanation}\n</thinking>\n\n"
                            f"### Production-Ready Code\n```{sample.language.value}\n{sample.solution_code.strip()}\n```\n\n"
                            f"### Unit Test Suite\n```{sample.language.value}\n{sample.test_code.strip()}\n```"
                        )
                        ok, s_id = harvester.harvest(sample.instruction, formatted_assistant, res_dummy, source="self_play")
                        if ok:
                            added += 1
                            print(f"  [+] Harvested [{added}/{count}]: {sample.instruction[:55]}...")
                print(f"[*] Self-play complete! Added {added}/{count} verified challenges to experience pool.\n")
                continue

            if user_input.lower().startswith("/dpo_gen"):
                count = 5
                parts = user_input.split()
                if len(parts) > 1 and parts[1].isdigit():
                    count = min(30, int(parts[1]))
                print(f"\n[*] Running Synthetic DPO Preference Generator ({count} pairs)...")
                dpo_gen = SyntheticDPOGenerator(harvester=harvester)
                added = dpo_gen.generate_batch(count=count)
                print(f"[*] DPO generation complete! Added {added} pairs to live_dpo_pairs.jsonl.\n")
                continue

            if user_input.lower() in ["/learn", "learn"]:
                stats = harvester.get_stats()
                if stats["pending_gold_samples"] == 0 and stats.get("pending_dpo_pairs", 0) == 0:
                    print("\n[*] No pending verified samples or DPO pairs in experience pool. Chat more, run '/selfplay 5', or run '/dpo_gen 5' first.\n")
                    continue

                print("\n===========================================================")
                print("       APEX CODER - INCREMENTAL CONTINUAL LEARNING")
                print("===========================================================")
                print("[1/4] Unloading inference model from GPU memory...", end=" ", flush=True)
                vram_before = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                del model
                del tokenizer
                model, tokenizer = None, None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                vram_after = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                freed = max(0.0, vram_before - vram_after)
                print(f"Done! (GPU VRAM: {freed:.2f} GB freed)", flush=True)

                print("[2/4] Executing Subprocess-Isolated Micro-Learning...", flush=True)
                success, new_adapter, msg = execute_continual_learning(min_samples=1)
                print(f"      Result: {msg}")

                print("[3/4] Reloading active LoRA adapter into GPU memory...", flush=True)
                model, tokenizer = load_chat_model(lora_adapter_dir=str(new_adapter))

                print("[4/4] Continual learning cycle complete! Ready for chat.\n")
                print("=" * 60 + "\n", flush=True)
                continue

            messages.append({"role": "user", "content": user_input})
            final_response = handle_generation_and_verification(
                model, tokenizer, messages, verifier, harvester, user_input
            )
            # Store clean response without thinking tags into long-term history
            clean_response = strip_thinking_tags(final_response)
            messages.append({"role": "assistant", "content": clean_response})

            # Process sliding window FIFO and rolling memory summarization
            messages, cur_tok, max_tok, pruned = budget_mgr.process_and_prune_history(
                messages, tokenizer, SYSTEM_PROMPT
            )
            if pruned:
                print(f"[*] [Context FIFO & Memory Sync] Compressed {cur_tok:,}/{max_tok:,} tokens (Active turns: {(len(messages)-1)//2}).", flush=True)

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"\n[Error] {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apex Coder Interactive CLI Chat")
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter directory (defaults to active adapter)")
    parser.add_argument("--prompt", type=str, default=None, help="Run single prompt test")
    args = parser.parse_args()

    model, tokenizer = load_chat_model(lora_adapter_dir=args.adapter)
    chat_loop(model, tokenizer, prompt_once=args.prompt)
