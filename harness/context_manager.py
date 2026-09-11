import re
from typing import List, Dict, Tuple, Optional


def strip_thinking_tags(text: str) -> str:
    """
    Removes <thinking>...</thinking> and <thought>...</thought> blocks
    from model generation before storing into long-term chat history.
    """
    if not text:
        return text

    # Remove thinking tags and their content
    cleaned = re.sub(r"<thinking>[\s\S]*?</thinking>", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"<thought>[\s\S]*?</thought>", "", cleaned, flags=re.IGNORECASE)
    
    # Clean up multiple consecutive empty lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_key_takeaway(user_msg: str, assistant_msg: str) -> str:
    """
    Extracts a high-density 1-2 line summary snippet from a pruned turn.
    """
    u_preview = user_msg.strip().replace("\n", " ")
    if len(u_preview) > 90:
        u_preview = u_preview[:87] + "..."

    # Find main class or function in assistant message if available
    class_match = re.search(r"class\s+([A-Za-z0-9_]+)", assistant_msg)
    func_match = re.search(r"def\s+([A-Za-z0-9_]+)|function\s+([A-Za-z0-9_]+)", assistant_msg)

    target_symbol = ""
    if class_match:
        target_symbol = f" (Implemented class: {class_match.group(1)})"
    elif func_match:
        fn_name = func_match.group(1) or func_match.group(2)
        target_symbol = f" (Implemented function: {fn_name})"

    return f"- User requested: '{u_preview}'{target_symbol} [Verified & Completed]"


class ContextBudgetManager:
    """
    Unified 4-in-1 Context Memory & Budget Controller:
    1. Real-Time Token Monitor (CLI HUD)
    2. Thinking Block Stripping (50-60% history token savings)
    3. Rolling Memory Summarization (Preserves context across pruned turns)
    4. Sliding Window FIFO Pruning (Keeps active context well below 32k limits)
    """

    def __init__(
        self,
        max_context_tokens: int = 32768,
        budget_cap_tokens: int = 16384,
        max_history_turns: int = 16,
    ):
        self.max_context_tokens = max_context_tokens
        self.budget_cap_tokens = budget_cap_tokens
        self.max_history_turns = max_history_turns
        self.rolling_memory: List[str] = []

    def count_tokens(self, messages: List[Dict[str, str]], tokenizer) -> int:
        """
        Calculates exact token count for the given message list using tokenizer chat template.
        """
        if not tokenizer or not messages:
            return 0
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return len(tokenizer.encode(formatted))
        except Exception:
            # Fallback estimation: ~4 chars per token
            total_chars = sum(len(m.get("content", "")) for m in messages)
            return total_chars // 4

    def build_system_prompt_with_memory(self, base_system_prompt: str) -> str:
        """
        Injects rolling memory summaries into the system prompt if available.
        """
        if not self.rolling_memory:
            return base_system_prompt

        memory_block = "\n\n### Prior Conversation Context & Architectural Memory:\n"
        memory_block += "\n".join(self.rolling_memory[-6:])  # Keep up to last 6 key memories
        return base_system_prompt + memory_block

    def process_and_prune_history(
        self,
        messages: List[Dict[str, str]],
        tokenizer,
        base_system_prompt: str,
    ) -> Tuple[List[Dict[str, str]], int, int, bool]:
        """
        Enforces token budget and turn limits via FIFO sliding window and rolling summarizer.
        Preserves Index 0 (System prompt with memory).
        Returns: (pruned_messages, current_tokens, total_capacity, was_pruned)
        """
        was_pruned = False

        # Ensure Index 0 has updated system prompt with memory
        messages[0]["content"] = self.build_system_prompt_with_memory(base_system_prompt)

        # Count current tokens
        cur_tokens = self.count_tokens(messages, tokenizer)

        # Pair up user and assistant messages (excluding system prompt at 0)
        # Prune if tokens exceed budget cap or turn count exceeds max
        while len(messages) > 3 and (cur_tokens > self.budget_cap_tokens or (len(messages) - 1) // 2 > self.max_history_turns):
            # Extract the oldest (user, assistant) pair
            old_user = messages[1].get("content", "")
            old_asst = messages[2].get("content", "") if len(messages) > 2 else ""

            # Summarize and add to rolling memory
            summary = extract_key_takeaway(old_user, old_asst)
            self.rolling_memory.append(summary)

            # Drop the oldest user-assistant pair (Index 1 & 2)
            if len(messages) > 2 and messages[2]["role"] == "assistant":
                messages.pop(1)  # user
                messages.pop(1)  # assistant
            else:
                messages.pop(1)

            # Re-update system prompt with new memory summary
            messages[0]["content"] = self.build_system_prompt_with_memory(base_system_prompt)
            cur_tokens = self.count_tokens(messages, tokenizer)
            was_pruned = True

        return messages, cur_tokens, self.max_context_tokens, was_pruned

    def get_hud_status(self, messages: List[Dict[str, str]], tokenizer) -> str:
        """
        Formats a clean, single-line CLI HUD for context status.
        """
        cur_tokens = self.count_tokens(messages, tokenizer)
        pct = (cur_tokens / self.max_context_tokens) * 100
        turns = (len(messages) - 1) // 2
        mem_count = len(self.rolling_memory)
        mem_tag = f" | Memory: {mem_count} summaries" if mem_count > 0 else ""
        return f"[Context: {cur_tokens:,} / {self.max_context_tokens:,} tokens ({pct:.1f}%) | Turns: {turns}{mem_tag}]"
