"""Standalone HTTP adapter for CMN-C1-113."""

import logging
import os
import re
import secrets as _secrets_module
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any, cast
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from framework.utils.config_loader import load_config
from shared.secrets import factory as secrets_factory
from shared.secrets.inmemory_provider import InMemoryProvider
from src.graph.graph import VoiceFAQGraph
from src.nodes.tts_node import AUDIO_OUTPUT_DIR
from src.services.audio_files import DEFAULT_AUDIO_EXTENSIONS, validate_audio_bytes
from src.services.input_guidance import build_input_guidance

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
_config = load_config(str(_CONFIG_PATH)) if _CONFIG_PATH.exists() else {}

_configured_secrets_provider = secrets_factory(namespace="cmn", agent_name="cmn-c1-113")
_azure_secret_keys = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
)
_secrets_provider = InMemoryProvider(
    {
        key: value
        for key in _azure_secret_keys
        if (value := os.environ.get(key) or _configured_secrets_provider.get(key)) is not None
    },
    namespace="cmn",
    agent_name="cmn-c1-113",
)

agent = VoiceFAQGraph(config=_config)

_AUDIO_OUTPUT_TTL_SECONDS = int(_config.get("audio_output_ttl_seconds", 900))
_AUDIO_OUTPUT_ID_RE = re.compile(r"^[a-f0-9]{32}$")

# --- Audio transcription (STT) config -- consumed here, ahead of agent.invoke().
# ADR-005 forbids raw audio bytes in graph state (must stay msgpack-serializable),
# so bytes are read, validated, and transcribed to text entirely in this module;
# only the resulting transcript string ever reaches the graph.
_STT_MODEL = _config.get("stt_model", "base")
_AUDIO_ALLOWED_EXTENSIONS = (
    tuple(
        f".{str(extension).lower().lstrip('.')}"
        for extension in _config.get("audio_allowed_extensions", ["wav", "mp3", "ogg", "flac", "m4a"])
    )
    or DEFAULT_AUDIO_EXTENSIONS
)
_AUDIO_MAX_SIZE_BYTES = int(_config.get("audio_max_file_size_mb", 20)) * 1024 * 1024
_AUDIO_TRANSCRIBE_SCRATCH_DIR = _config.get("audio_transcribe_scratch_dir", "/tmp/cmn-c1-113/transcribe")
_AUDIO_CHUNK_SIZE = 1024 * 1024

try:
    import whisper as _whisper

    _WHISPER_AVAILABLE = True
except ImportError:
    _whisper = None
    _WHISPER_AVAILABLE = False

_WHISPER_MODELS: dict[str, Any] = {}
_WHISPER_MODEL_LOCK = Lock()


def _get_whisper_model(model_name: str) -> Any:
    """Load each non-secret Whisper model at most once per process."""
    with _WHISPER_MODEL_LOCK:
        if model_name not in _WHISPER_MODELS:
            model_dir = os.environ.get("WHISPER_MODEL_DIR")
            kwargs = {"download_root": model_dir} if model_dir else {}
            _WHISPER_MODELS[model_name] = _whisper.load_model(model_name, **kwargs)
        return _WHISPER_MODELS[model_name]


async def _read_upload_bounded(file: UploadFile, *, max_size_bytes: int) -> bytes | None:
    """Read an upload in chunks, aborting as soon as it exceeds max_size_bytes.

    Returns None (instead of the full buffer) when the size limit is exceeded,
    so an oversized upload is never fully buffered in memory.
    """
    buffer = bytearray()
    while True:
        chunk = await file.read(_AUDIO_CHUNK_SIZE)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > max_size_bytes:
            return None
    return bytes(buffer)


def _transcribe(audio_bytes: bytes, suffix: str) -> tuple[str | None, dict[str, Any] | None]:
    """Transcribe validated audio bytes to text.

    Returns (transcript, None) on success, or (None, guidance_response) on
    any failure -- guidance_response is a build_input_guidance() dict ready
    to return directly to the caller.
    """
    if not _WHISPER_AVAILABLE:
        return None, build_input_guidance(
            "Audio transcription is temporarily unavailable. Please enter the FAQ question as text."
        )

    Path(_AUDIO_TRANSCRIBE_SCRATCH_DIR).mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=_AUDIO_TRANSCRIBE_SCRATCH_DIR, suffix=suffix, delete=True) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file.flush()
            model = _get_whisper_model(_STT_MODEL)
            result = model.transcribe(tmp_file.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("cmn-c1-113: audio transcription failed: %s", type(exc).__name__)
        return None, build_input_guidance(
            "The audio could not be transcribed. Please retry with a clear recording or enter the question as text."
        )

    transcript = str(result.get("text", "")).strip()
    if not transcript:
        return None, build_input_guidance("No speech could be recognized in the audio file.")
    return transcript, None


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        agent.compile()
    except Exception as exc:
        logger.error("cmn-c1-113: agent.compile() failed: %s", exc)
        raise
    try:
        agent.provision_secrets(_secrets_provider)
    except Exception as exc:
        logger.error("cmn-c1-113: failed to provision secrets at startup: %s", exc)
    yield


app = FastAPI(title="VoiceActivatedFAQAgent", lifespan=_lifespan)


class InvokeRequest(BaseModel):
    """Public contract: a plain text question."""

    model_config = ConfigDict(extra="forbid")

    input: str = ""
    session_id: str = ""


def _bearer_matches(supplied: str, expected: str) -> bool:
    """Compare a supplied bearer token in constant time."""
    return _secrets_module.compare_digest(supplied.encode(), f"Bearer {expected}".encode())


def _resolve_standalone_trust(
    current: TrustLevel,
    authorization: str,
    invoke_auth_token: str | None,
    internal_runner_token: str | None,
) -> TrustLevel:
    """Resolve standalone bearer authentication without changing middleware trust."""
    if current is not TrustLevel.ANONYMOUS:
        return current
    if internal_runner_token and _bearer_matches(authorization, internal_runner_token):
        return TrustLevel.INTERNAL
    if invoke_auth_token and _bearer_matches(authorization, invoke_auth_token):
        return TrustLevel.VERIFIED_EXTERNAL
    if internal_runner_token or invoke_auth_token:
        raise HTTPException(status_code=401, detail="Token is invalid or expired.")
    return TrustLevel.ANONYMOUS


def _cleanup_stale_audio() -> None:
    """Best-effort cleanup of generated WAV files after their configured TTL."""
    output_directory = Path(AUDIO_OUTPUT_DIR)
    try:
        now = time.time()
        for entry in output_directory.iterdir():
            try:
                if entry.is_file() and now - entry.stat().st_mtime > _AUDIO_OUTPUT_TTL_SECONDS:
                    entry.unlink()
            except OSError:
                continue
    except FileNotFoundError:
        return


def _authenticate(request: Request) -> TrustLevel:
    return _resolve_standalone_trust(
        getattr(request.state, "trust_level", TrustLevel.ANONYMOUS),
        request.headers.get("authorization", ""),
        os.environ.get("INVOKE_AUTH_TOKEN"),
        os.environ.get("STG_INTERNAL_RUNNER_TOKEN"),
    )


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> dict[str, Any]:
    trust_level = _authenticate(request)
    _cleanup_stale_audio()
    ctx = InvocationContext(
        session_id=req.session_id or str(uuid4()),
        caller_trust_level=trust_level,
        caller_id=getattr(request.state, "caller_id", ""),
    )
    with bound_secrets(_secrets_provider):
        return cast("dict[str, Any]", agent.invoke(user_input=req.input, ctx=ctx))


@app.post("/invoke/audio")
async def invoke_audio(
    request: Request,
    audio: UploadFile = File(...),
    session_id: str = Form(""),
) -> dict[str, Any]:
    trust_level = _authenticate(request)

    audio_bytes = await _read_upload_bounded(audio, max_size_bytes=_AUDIO_MAX_SIZE_BYTES)
    if audio_bytes is None:
        return build_input_guidance("The audio file exceeds the configured size limit.")

    validation_error = validate_audio_bytes(
        audio.filename or "",
        len(audio_bytes),
        allowed_extensions=_AUDIO_ALLOWED_EXTENSIONS,
        max_size_bytes=_AUDIO_MAX_SIZE_BYTES,
    )
    if validation_error:
        return build_input_guidance(validation_error)

    suffix = PurePosixPath(audio.filename or "").suffix.lower()
    transcript, guidance = _transcribe(audio_bytes, suffix)
    if guidance is not None:
        return guidance

    _cleanup_stale_audio()
    ctx = InvocationContext(
        session_id=session_id or str(uuid4()),
        caller_trust_level=trust_level,
        caller_id=getattr(request.state, "caller_id", ""),
    )
    with bound_secrets(_secrets_provider):
        return cast(
            "dict[str, Any]",
            agent.invoke(
                user_input=transcript,
                input_context={"input_mode": "audio", "audio_response_requested": True},
                ctx=ctx,
            ),
        )


@app.get("/audio/{audio_output_id}")
def get_audio(audio_output_id: str, request: Request) -> FileResponse:
    """Return an authenticated, generated WAV file by opaque output id."""
    _authenticate(request)
    if not _AUDIO_OUTPUT_ID_RE.fullmatch(audio_output_id):
        raise HTTPException(status_code=400, detail="Invalid audio output id.")
    _cleanup_stale_audio()
    candidate = Path(AUDIO_OUTPUT_DIR).resolve() / f"{audio_output_id}.wav"
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Audio output was not found or has expired.")
    return FileResponse(candidate, media_type="audio/wav", filename=f"{audio_output_id}.wav")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "cmn-c1-113"}
