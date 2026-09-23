"""Conversation-safe guidance for unsupported or invalid FAQ requests."""

from typing import Any

from framework.schemas.agent_status import AgentStatus


def build_input_guidance(reason: str, *, heading: str = "Customer FAQ question required") -> dict[str, Any]:
    """Build a successful Markdown guidance response instead of routing to an answer.

    Finding: every caller used to get this same fixed heading regardless of
    *why* no answer was generated. GeneratorNode calls this both for
    genuine input problems (empty question, over-length, prompt-injection
    pattern, off-topic message) and for temporary system conditions (LLM
    provider unavailable, timeout, capacity exhausted, malformed/empty LLM
    response) -- situations with nothing in common except "no answer was
    produced". A fixed "Customer FAQ question required" heading on a
    provider outage reads as "you asked the wrong kind of question" even
    though the actual cause was the LLM call failing, making a real
    infrastructure problem indistinguishable from a user input mistake.
    `heading` lets each call site say what actually happened; the default
    preserves the original wording for genuine input-validation cases.
    """
    output = (
        f"# {heading}\n\n"
        f"{reason}\n\n"
        "Please enter a clear customer-service question about an order, payment, "
        "account, refund, delivery, or service information.\n\n"
        "## Example\n\n"
        "> I was charged twice for my order. How can I request a refund?"
    )
    return {
        "input_validation_failed": "true",
        "answer_text": None,
        "answer_language": None,
        "audio_output_path": None,
        "audio_output_id": None,
        "audio_duration_sec": None,
        "formatted_output": output,
        "error_message": None,
        "status": AgentStatus.SUCCESS.value,
    }
