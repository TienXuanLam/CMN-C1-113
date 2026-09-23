"""Generate a grounded FAQ response through the AgentCore Azure OpenAI client."""

from __future__ import annotations

import json
from collections.abc import Callable
from queue import Empty, Queue
from threading import BoundedSemaphore, Thread
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from shared.services.llm.azure_openai_client import AzureOpenAIClient
from shared.services.llm.base_llm import BaseLLM
from shared.utils.audit_logger import emit_trace_event

from src.nodes.security_node import _INJECTION_PATTERNS
from src.schemas.state import VoiceFAQState
from src.services.input_guidance import build_input_guidance

try:
    from langdetect import detect as langdetect_detect

    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

_MAX_TRANSCRIPT_CHARS = 2000
_MAX_CONTEXT_CHARS = 4000
_LLM_CALL_SLOTS = BoundedSemaphore(value=4)


class _ChatCompletionsAdapter:
    """Adapts a raw OpenAI-style `client.chat.completions.create(...)`
    object (as returned by `openai_client_factory`) to the `BaseLLM.complete()`
    shape `_complete_with_deadline()` expects, so the client_factory
    injection path shares the same thread-deadline and capacity-semaphore
    protection as the default AzureOpenAIClient path."""

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        response = self._client.chat.completions.create(model=self._model, messages=messages)
        return {"content": str(response.choices[0].message.content or "")}


class _CapacityExhaustedError(Exception):
    """All _LLM_CALL_SLOTS are held; raised instantly, never after a wait.

    Kept distinct from TimeoutError so the caller can tell "no slot was
    free right now" (every slot held, possibly by a leaked worker thread
    still stuck inside a prior llm.complete() that never returned) apart
    from "a slot was free, the call started, and it ran past
    _LLM_TIMEOUT_SECONDS" -- different causes, different guidance.
    """


_SYSTEM_PROMPT_BASE = """You are a multilingual customer-service FAQ assistant.

Decide first whether the user supplied a meaningful customer-service FAQ
question. Random text, greetings without a question, coding tasks, general
knowledge, and unrelated requests are not business-relevant.

Answer using the supplied FAQ context when it is relevant. If no matching FAQ
context is available, provide only safe, general next-step guidance and clearly
state that the configured FAQ did not contain a definitive answer. Do not invent
company policies, prices, deadlines, eligibility, or guarantees.

Return JSON only:
{
  "business_relevant": true or false,
  "answer": "concise answer in the same language as the question"
}
"""

# Finding: docs/02_design.md, docs/03_test_spec.md (PB-9), and
# docs/07_operation_guide.md all document a low-confidence disclaimer --
# "when retrieval_score_max < 0.45 ... GeneratorNode injects a
# disclaimer" -- and a disclaimer_injected field on the generator_complete
# trace event. Neither existed: retrieval_score_max was never read here,
# _SYSTEM_PROMPT was a fixed module constant, and generator_complete's
# payload had no such field. Appended (not substituted) so the base
# instructions -- including the JSON-only response contract -- are never
# altered by the low-confidence case.
_LOW_CONFIDENCE_DISCLAIMER = (
    "\n\nThe retrieved FAQ context has low relevance to this question. Clearly tell the "
    "user the configured FAQ did not contain a definitive answer, and keep the response "
    "to safe, general next-step guidance only."
)


class GeneratorNode(FunctionNode):
    """Generate answer text from transcript and retrieved FAQ context."""

    required_trust_level = TrustLevel.ANONYMOUS

    def __init__(
        self,
        llm_model: str = "",
        client_factory: Callable[[str], Any] | None = None,
        llm_client: BaseLLM | None = None,
        timeout_s: float = 25.0,
        max_retry: int = 0,
        retrieval_score_threshold: float = 0.45,
    ) -> None:
        self._llm_model = llm_model
        self._client_factory = client_factory
        self._llm = llm_client
        self._retrieval_score_threshold = retrieval_score_threshold
        # Finding: previously hard-coded (timeout=25.0, max_retries=0) --
        # config.yaml's timeout_s/max_retry were never read, the same
        # dead-config bug found elsewhere in the fleet's equivalent nodes.
        # max_retries=0 meant a single
        # transient provider hiccup was never retried by the SDK itself; the
        # thread deadline below still bounds worst-case latency per attempt,
        # so allowing the SDK's own retries no longer risks an unbounded hang.
        self._timeout_s = timeout_s
        self._max_retry = max_retry

    def execute(self, state: VoiceFAQState) -> dict[str, Any]:
        emit_trace_event(
            event_type="generator_start",
            payload={
                "transcript_chars": len(state.get("transcript") or ""),
                "retrieved_chunks_chars": len(state.get("retrieved_chunks") or ""),
            },
            state=state,
        )
        if state.get("input_validation_failed") == "true" or state.get("error_message"):
            return {}

        transcript = str(state.get("transcript") or "").strip()
        if not transcript:
            return build_input_guidance("The question is empty.")
        if len(transcript) > _MAX_TRANSCRIPT_CHARS:
            return build_input_guidance(f"The question exceeds {_MAX_TRANSCRIPT_CHARS} characters.")

        context_str = str(state.get("retrieved_chunks") or "[]")[:_MAX_CONTEXT_CHARS]
        for value, label in ((transcript, "question"), (context_str, "FAQ context")):
            if any(pattern.search(value) for pattern in _INJECTION_PATTERNS):
                return build_input_guidance(f"The {label} contains unsupported instruction-like content.")

        retrieval_score_max = state.get("retrieval_score_max")
        disclaimer_injected = retrieval_score_max is not None and retrieval_score_max < self._retrieval_score_threshold
        system_prompt = _SYSTEM_PROMPT_BASE + (_LOW_CONFIDENCE_DISCLAIMER if disclaimer_injected else "")

        user_message = f"Customer question:\n{transcript}\n\nConfigured FAQ context:\n{context_str}"
        try:
            raw_text = self._invoke_llm(state, user_message, system_prompt)
        except Exception as exc:  # noqa: BLE001
            # Finding: this used to return the same "Customer FAQ question
            # required" heading as a genuine input problem (empty question,
            # over-length, injection pattern) for every possible failure
            # here -- a real LLM/provider outage looked identical to a user
            # asking the wrong kind of question in every bug report. A
            # direct call to the same Azure OpenAI deployment with the same
            # prompt/context succeeded in ~2s, so a failure at this call
            # site is a genuine temporary system condition, not something
            # caused by the question itself; distinguishing capacity
            # exhaustion (instant, no wait) from a real per-call timeout
            # (waited the full deadline) from any other provider failure
            # lets a support investigation start from the right cause
            # instead of re-deriving it from scratch each time.
            # Only the exception's class name (e.g. "ConnectError",
            # "APITimeoutError") is surfaced to the UI, never str(exc) --
            # the message text can contain internal details (proxy host,
            # endpoint URL, stack fragments) that must not leave the
            # backend. The class name alone is enough to tell network vs
            # timeout vs auth apart without exposing infrastructure.
            reference = self._reference_note(state)
            diagnostic = f"Diagnostic: {type(exc).__name__}."
            if isinstance(exc, _CapacityExhaustedError):
                emit_trace_event(
                    event_type="generator_capacity_exhausted",
                    payload={"available_slots": 0},
                    state=state,
                )
                return build_input_guidance(
                    f"The FAQ answer service is at capacity right now. This is not caused by your question "
                    f"— please try again in a moment. {reference} {diagnostic}",
                    heading="Service busy",
                )
            if self._is_timeout_error(exc):
                emit_trace_event(
                    event_type="generator_timeout",
                    payload={"timeout_seconds": self._timeout_s, "error_type": type(exc).__name__},
                    state=state,
                )
                return build_input_guidance(
                    f"The answer could not be generated in time. This is usually temporary and not caused by "
                    f"your question — please try again. {reference} {diagnostic}",
                    heading="Answer generation timed out",
                )
            emit_trace_event(
                event_type="generator_provider_unavailable",
                payload={"error_type": type(exc).__name__},
                state=state,
            )
            return build_input_guidance(
                f"The FAQ answer service is temporarily unavailable. This is not caused by your question "
                f"— please try again shortly. {reference} {diagnostic}",
                heading="Service temporarily unavailable",
            )

        parsed = self._parse_json(raw_text)
        if parsed is not None:
            if parsed.get("business_relevant") is False:
                return build_input_guidance(
                    "The message is not a customer-service FAQ question supported by this agent."
                )
            answer_text = str(parsed.get("answer") or "").strip()
        else:
            answer_text = raw_text.strip()
        if not answer_text:
            # A blank/unparseable LLM response is also a provider-side
            # anomaly, not a question the user asked incorrectly.
            return build_input_guidance(
                f"The answer service returned an unexpected response. This is not caused by your question "
                f"— please try again. {self._reference_note(state)}",
                heading="Service temporarily unavailable",
            )

        answer_language = "unknown"
        if _LANGDETECT_AVAILABLE:
            try:
                answer_language = langdetect_detect(answer_text)
            except Exception:  # noqa: BLE001
                answer_language = "unknown"

        emit_trace_event(
            event_type="generator_complete",
            payload={
                "answer_chars": len(answer_text),
                "answer_language": answer_language,
                "disclaimer_injected": disclaimer_injected,
            },
            state=state,
        )
        return {"answer_text": answer_text, "answer_language": answer_language}

    def _invoke_llm(self, state: VoiceFAQState, user_message: str, system_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        # Finding: the client_factory path previously called
        # client.chat.completions.create(...) directly, bypassing both
        # _complete_with_deadline() (the thread-based timeout) and
        # _LLM_CALL_SLOTS (the capacity semaphore) that the default
        # AzureOpenAIClient path below relies on. openai_client_factory is
        # a real config.yaml key, not test-only, so that gap was reachable
        # in production and could hang the calling thread indefinitely.
        # Wrapping the factory-built client in a BaseLLM-shaped adapter
        # lets both paths share the same deadline/capacity protection.
        if self._client_factory is not None:
            client = self._client_factory("injected-test-client")
            llm: BaseLLM = _ChatCompletionsAdapter(client, self._llm_model)
        else:
            llm = self._llm or self._build_llm(state)
        response = self._complete_with_deadline(
            llm,
            messages,
            timeout_seconds=self._timeout_s,
        )
        return str(response.get("content") or "")

    @staticmethod
    def _reference_note(state: dict[str, Any]) -> str:
        reference = str(state.get("trace_id") or state.get("correlation_id") or "unavailable")
        return f"Reference: {reference}."

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        """Recognise provider/transport timeouts without coupling to one SDK."""
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            name = type(current).__name__.lower()
            message = str(current).lower()
            if "timeout" in name or "timed out" in message or "timeout" in message:
                return True
            current = current.__cause__ or current.__context__
        return False

    def _build_llm(self, state: VoiceFAQState) -> BaseLLM:
        ctx = InvocationContext.from_state(state)
        return AzureOpenAIClient(
            {
                "api_key": ctx.secrets.require("AZURE_OPENAI_API_KEY"),
                "azure_endpoint": ctx.secrets.require("AZURE_OPENAI_ENDPOINT"),
                "azure_deployment": ctx.secrets.require("AZURE_OPENAI_DEPLOYMENT"),
                "timeout": self._timeout_s,
                "max_retries": self._max_retry,
                "max_tokens": 900,
            }
        )

    @staticmethod
    def _complete_with_deadline(
        llm: BaseLLM, messages: list[dict[str, str]], *, timeout_seconds: float
    ) -> dict[str, Any]:
        if not _LLM_CALL_SLOTS.acquire(blocking=False):
            raise _CapacityExhaustedError("FAQ generation capacity is temporarily exhausted.")
        result_queue: Queue[tuple[bool, dict[str, Any] | Exception]] = Queue(maxsize=1)

        def _worker() -> None:
            try:
                result_queue.put((True, llm.complete(messages)))
            except Exception as exc:  # noqa: BLE001
                result_queue.put((False, exc))
            finally:
                _LLM_CALL_SLOTS.release()

        Thread(target=_worker, name="faq-azure-openai", daemon=True).start()
        try:
            succeeded, payload = result_queue.get(timeout=timeout_seconds)
        except Empty as exc:
            raise TimeoutError(f"FAQ generation exceeded {timeout_seconds:.1f}s.") from exc
        if not succeeded:
            if isinstance(payload, Exception):
                raise payload
            raise RuntimeError("FAQ generation failed without an exception payload.")
        if not isinstance(payload, dict):
            raise TypeError("LLM response must be a dictionary.")
        return payload

    @staticmethod
    def _parse_json(raw_text: str) -> dict[str, Any] | None:
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
