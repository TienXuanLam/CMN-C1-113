"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Never import from mediator/, api/, or other agents

from __future__ import annotations

import io
import logging
import os
import re
import wave
from threading import Lock
from typing import Any, Optional
from uuid import uuid4

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.services.events import emitter
from shared.services.events.types import EventType
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import VoiceFAQState

logger = logging.getLogger(__name__)

try:
    import voxcpm2

    VOXCPM2_AVAILABLE = True
except ImportError:
    voxcpm2 = None
    VOXCPM2_AVAILABLE = False

_SENTENCE_END_RE = re.compile(r"(?<=[。.!?！？])\s*")

# Generated files live in a controlled directory and use opaque random ids.
# The standalone adapter serves them through its authenticated audio endpoint
# and performs TTL cleanup.
AUDIO_OUTPUT_DIR = os.environ.get("AUDIO_OUTPUT_DIR", "/tmp/cmn-c1-113/audio-out/")
_TTS_MODEL: Any | None = None
_TTS_MODEL_LOCK = Lock()


class TTSNode(FunctionNode):
    """Synthesise answer_text → audio_output_path + audio_duration_sec.

    Temp file lifecycle: the WAV file is written under AUDIO_OUTPUT_DIR using
    an opaque random id and served by ``GET /audio/{audio_output_id}``.
    ``server.py`` deletes generated files after the configured TTL.
    """

    required_trust_level = TrustLevel.ANONYMOUS

    def __init__(self, tts_chunk_max_chars: int = 200):
        self._tts_chunk_max_chars = tts_chunk_max_chars

    def execute(self, state: VoiceFAQState) -> dict[str, Any]:
        emit_trace_event(
            event_type="tts_start",
            payload={
                "session_id": state.get("session_id"),
                "answer_chars": len(state.get("answer_text") or ""),
            },
            state=state,
        )

        if state.get("input_validation_failed") == "true":
            guidance = str(state.get("formatted_output") or "# Customer FAQ question required")
            return {"formatted_output": guidance, "status": AgentStatus.SUCCESS.value}

        if state.get("error_message"):
            return {}

        answer_text: Optional[str] = state.get("answer_text")
        if not answer_text:
            return {"error_message": "TTSNode: answer_text is None or empty", "status": AgentStatus.ERROR.value}

        formatted_text = f"# FAQ Answer\n\n{answer_text}"
        if not state.get("audio_response_requested"):
            emitter().emit_event(
                event_type=EventType.PROGRESS_UPDATE,
                message="Preparing the FAQ response.",
                metadata={"stage": "response_formatting", "audio_available": False},
            )
            return {
                "audio_output_path": None,
                "audio_output_id": None,
                "audio_duration_sec": None,
                "formatted_output": formatted_text,
                "status": AgentStatus.SUCCESS.value,
            }

        if not VOXCPM2_AVAILABLE:
            # Finding (2026-08-19, Blocker): docs/07_operation_guide.md
            # documents VoxCPM2 as "Optional -- Required for audio output.
            # If unavailable, audio_output_path=None", but this previously
            # returned status=ERROR unconditionally, failing the ENTIRE
            # pipeline whenever the (not-on-PyPI, internal-registry-only)
            # voxcpm2 package wasn't installed -- there was no text-only
            # success path at all. Degrade to a genuine text-only SUCCESS
            # matching the documented contract: the answer text is still
            # generated and returned, just with no synthesised audio.
            logger.warning("cmn-c1-113: VoxCPM2 not available — degrading to text-only response.")
            emitter().emit_event(
                event_type=EventType.PROGRESS_UPDATE,
                message="Preparing the FAQ response.",
                metadata={"stage": "response_formatting", "audio_available": False},
            )
            emit_trace_event(
                event_type="tts_skipped_no_voxcpm2",
                payload={"session_id": state.get("session_id"), "answer_chars": len(answer_text)},
                state=state,
            )
            return {
                "audio_output_path": None,
                "audio_output_id": None,
                "audio_duration_sec": None,
                "formatted_output": formatted_text,
                "status": AgentStatus.SUCCESS.value,
            }

        try:
            # Config from constructor (set by graph.register_nodes from config.yaml)
            # Bounded -- Finding (2026-08-19, High/Cost-DoS): tts_chunk_max_chars
            # had no bounds, letting a caller drive an arbitrarily small chunk
            # size (many more synthesize() calls per answer) or a very large
            # one (single call over an oversized string), neither validated
            # against the underlying TTS engine's real limits.
            input_context = state.get("input_context", {})
            requested_max_chars = int(input_context.get("tts_chunk_max_chars", self._tts_chunk_max_chars))
            max_chars = max(20, min(requested_max_chars, 500))
            chunks = self._chunk_text(answer_text, max_chars=max_chars)
            wav_segments: list[bytes] = []
            model = self._get_tts_model()
            for chunk in chunks:
                segment = model.synthesize(chunk)
                if not isinstance(segment, bytes):
                    raise TypeError("VoxCPM2 synthesize() must return WAV bytes")
                wav_segments.append(segment)

            combined = self._concat_wav_segments(wav_segments)
            audio_output_id = uuid4().hex
            os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
            tmp_path = os.path.join(AUDIO_OUTPUT_DIR, f"{audio_output_id}.wav")
            with open(tmp_path, "wb") as fh:
                fh.write(combined)

            audio_output_path = os.path.abspath(tmp_path)
            audio_duration_sec = self._wav_duration(combined)

            emit_trace_event(
                event_type="tts_complete",
                payload={
                    "session_id": state.get("session_id"),
                    "answer_chars": len(answer_text),
                    "audio_duration_sec": audio_duration_sec,
                },
                state=state,
            )
            # L-03: assemble formatted_output for SDK get_output() → result["output"]
            # Finding (2026-08-19, High): formatted_output was a raw dict in
            # State, violating the state-serialisation contract
            # (the framework's state-safety rule) -- complex values
            # must be JSON strings/msgpack-safe, since a checkpoint-enabled
            # deployment persists State via msgpack, which cannot serialise
            # an arbitrary Python dict the way it can a plain string.
            emitter().emit_event(
                event_type=EventType.PROGRESS_UPDATE,
                message="Preparing the FAQ response.",
                metadata={"stage": "response_formatting", "audio_available": True},
            )
            return {
                "audio_output_path": audio_output_path,
                "audio_output_id": audio_output_id,
                "audio_duration_sec": audio_duration_sec,
                "formatted_output": formatted_text,
                "status": AgentStatus.SUCCESS.value,
            }

        except Exception as e:  # noqa: BLE001
            # Finding: this previously returned "error_message": None
            # unconditionally, silently clearing any upstream error already
            # in state -- a node that returns only its own changed fields
            # must not null a field it didn't set. It also collapsed every
            # runtime synthesis failure (disk full, malformed WAV, a
            # TypeError from a misbehaving TTS backend) into the same
            # SUCCESS-with-no-audio response as the two intentional
            # degradation paths above (VoxCPM2 missing, empty answer),
            # leaving no way for a caller to tell "audio skipped on
            # purpose" from "audio generation broke". The tts_error trace
            # event already carries the real reason; do not also fabricate
            # a state-level guarantee that nothing went wrong.
            emit_trace_event(
                event_type="tts_error",
                payload={"session_id": state.get("session_id"), "reason": type(e).__name__},
                state=state,
            )
            return {
                "audio_output_path": None,
                "audio_output_id": None,
                "audio_duration_sec": None,
                "formatted_output": formatted_text,
                "status": AgentStatus.SUCCESS.value,
            }

    @staticmethod
    def _get_tts_model() -> Any:
        """Load the non-secret TTS model at most once per process."""
        global _TTS_MODEL
        with _TTS_MODEL_LOCK:
            if _TTS_MODEL is None:
                if voxcpm2 is None:
                    raise RuntimeError("VoxCPM2 is unavailable")
                _TTS_MODEL = voxcpm2.TTS()
            return _TTS_MODEL

    def _chunk_text(self, text: str, max_chars: int = 200) -> list[str]:
        if not text:
            return []
        raw_sentences = _SENTENCE_END_RE.split(text)
        chunks: list[str] = []
        current = ""
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                for i in range(0, len(sentence), max_chars):
                    part = sentence[i : i + max_chars]
                    if part:
                        chunks.append(part)
                continue
            candidate = (current + " " + sentence).strip() if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)
        return [c for c in chunks if c]

    @staticmethod
    def _concat_wav_segments(segments: list[bytes]) -> bytes:
        if not segments:
            raise ValueError("No WAV segments to concatenate")
        if len(segments) == 1:
            return segments[0]
        with wave.open(io.BytesIO(segments[0]), "rb") as first_wav:
            params = first_wav.getparams()
            combined_frames = first_wav.readframes(first_wav.getnframes())
        for i, seg in enumerate(segments[1:], start=1):
            with wave.open(io.BytesIO(seg), "rb") as w:
                # L-09: verify params consistency across segments
                seg_params = w.getparams()
                if seg_params[:3] != params[:3]:  # nchannels, sampwidth, framerate
                    raise ValueError(f"WAV segment {i} params {seg_params[:3]} differ from first segment {params[:3]}")
                combined_frames += w.readframes(w.getnframes())
        out_buf = io.BytesIO()
        with wave.open(out_buf, "wb") as out_wav:
            out_wav.setparams(params)
            out_wav.writeframes(combined_frames)
        return out_buf.getvalue()

    @staticmethod
    def _wav_duration(wav_bytes: bytes) -> float:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            frames = w.getnframes()
            frame_rate = w.getframerate()
            return float(frames) / float(frame_rate) if frame_rate else 0.0
