"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - S-4: transcript content NOT logged — only char_count emitted
#  - Never import from mediator/, api/, or other agents
#
# Audio transcription happens in src/api/server.py, ahead of agent.invoke() —
# ADR-005 forbids raw audio bytes in state (msgpack-serializable only), so by
# the time a request reaches this node it is always already-transcribed text.

from __future__ import annotations

from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.trust_level import TrustLevel
from shared.services.events import emitter
from shared.services.events.types import EventType
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import VoiceFAQState
from src.services.input_guidance import build_input_guidance


class STTNode(FunctionNode):
    """Intake node. Passes the already-resolved text question through as the transcript."""

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: VoiceFAQState) -> dict[str, Any]:
        transient_reset: dict[str, Any] = {
            "input_validation_failed": None,
            "input_mode": None,
            "audio_response_requested": None,
            "transcript": None,
            "stt_language": None,
            "stt_confidence": None,
            "retrieved_chunks": None,
            "retrieval_score_max": None,
            "answer_text": None,
            "answer_language": None,
            "audio_output_path": None,
            "audio_output_id": None,
            "audio_duration_sec": None,
            "formatted_output": None,
            "error_message": None,
        }
        session_id: str = state.get("session_id", "")
        input_context = state.get("input_context") or {}
        text_input: str = (state.get("user_input") or state.get("text_input") or "").strip()
        input_mode = input_context.get("input_mode", "text")
        audio_response_requested = bool(input_context.get("audio_response_requested"))

        emit_trace_event(
            event_type="stt_start",
            payload={"session_id": session_id, "has_text_input": bool(text_input)},
            state=state,
        )

        if text_input:
            emitter().emit_event(
                event_type=EventType.PROGRESS_UPDATE,
                message="Validating the FAQ question.",
                metadata={"stage": "input_validation"},
            )
            return {
                **transient_reset,
                "input_mode": input_mode,
                "audio_response_requested": audio_response_requested,
                **self._text_passthrough(state, text_input, session_id),
            }

        emit_trace_event(
            event_type="stt_error",
            payload={"session_id": session_id, "reason": "no input provided"},
            state=state,
        )
        return {**transient_reset, **build_input_guidance("The question is empty.")}

    def _text_passthrough(self, state: dict[str, Any], text_input: str, session_id: str) -> dict[str, Any]:
        transcript = text_input.strip()
        emit_trace_event(
            event_type="stt_complete",
            payload={
                "session_id": session_id,
                "mode": "text_passthrough",
                "transcript_char_count": len(transcript),
            },
            state=state,
        )
        return {"transcript": transcript, "stt_language": "text_passthrough", "stt_confidence": 1.0}
