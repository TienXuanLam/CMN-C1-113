# Proof-of-Boundary: HTTP layer — CMN-C1-113 server.py two-endpoint contract.
#
# First HTTP-layer test in this repo: uses fastapi.testclient.TestClient
# against the real `app` object exported by src/api/server.py. Follows the
# same import/reload + monkeypatch-env-vars pattern already established by
# test_server_llm_injection.py (see its module comment) rather than
# inventing a new mocking approach.
#
# POST /invoke/audio contract (src/api/server.py):
#   - oversized upload -> build_input_guidance() "exceeds the configured
#     size limit" (still HTTP 200 / status=success -- see
#     src/services/input_guidance.py's build_input_guidance() docstring:
#     it is a *successful* Markdown guidance response, not an error).
#   - disallowed extension -> validate_audio_bytes() unsupported-format
#     guidance.
#   - whisper not installed -> _transcribe()'s "temporarily unavailable"
#     guidance, with no real audio bytes/model needed.
#
# POST /invoke contract: JSON {"input": str, "session_id": str},
# extra="forbid" on InvokeRequest -> unknown fields (e.g. the old
# audio_input_path) are rejected with HTTP 422.

import importlib

from fastapi.testclient import TestClient


def _reloaded_server():
    """Import (or re-import) src.api.server fresh, matching
    test_server_llm_injection.py's established pattern."""
    import src.api.server as server

    importlib.reload(server)
    return server


class TestInvokeAudioSizeLimit:
    def test_oversized_upload_returns_guidance_not_error(self, monkeypatch):
        server = _reloaded_server()
        # Force a tiny limit instead of buffering a real oversized file.
        monkeypatch.setattr(server, "_AUDIO_MAX_SIZE_BYTES", 100)

        client = TestClient(server.app)
        oversized = b"RIFF....WAVEfmt " + b"\x00" * 200
        response = client.post(
            "/invoke/audio",
            files={"audio": ("question.wav", oversized, "audio/wav")},
            data={"session_id": "test-session-oversized"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["input_validation_failed"] == "true"
        assert "exceeds the configured size limit" in body["formatted_output"]


class TestInvokeAudioUnsupportedFormat:
    def test_disallowed_extension_returns_guidance(self, monkeypatch):
        server = _reloaded_server()

        client = TestClient(server.app)
        response = client.post(
            "/invoke/audio",
            files={"audio": ("test.txt", b"not audio", "text/plain")},
            data={"session_id": "test-session-badext"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["input_validation_failed"] == "true"
        assert "not supported" in body["formatted_output"]


class TestInvokeAudioWhisperUnavailable:
    def test_whisper_unavailable_returns_guidance_without_real_audio(self, monkeypatch):
        server = _reloaded_server()
        monkeypatch.setattr(server, "_WHISPER_AVAILABLE", False)

        client = TestClient(server.app)
        response = client.post(
            "/invoke/audio",
            files={"audio": ("question.wav", b"tiny-fake-audio-bytes", "audio/wav")},
            data={"session_id": "test-session-nowhisper"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["input_validation_failed"] == "true"
        assert "temporarily unavailable" in body["formatted_output"]


class TestInvokeJsonText:
    def test_valid_json_input_is_accepted(self, monkeypatch):
        # Follow test_server_llm_injection.py's TestServerBootsWithoutAzureKeys
        # pattern: no Azure key provisioned, so GeneratorNode's own
        # except-branch converts the failed LLM call into a
        # build_input_guidance() response -- the request still completes
        # end to end without real credentials.
        for key in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"):
            monkeypatch.delenv(key, raising=False)
        server = _reloaded_server()
        server.agent.compile()

        client = TestClient(server.app)
        response = client.post(
            "/invoke",
            json={"input": "店の営業時間を教えてください", "session_id": "test-session-json"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] in ("success", "error")

    def test_unknown_extra_field_is_rejected(self, monkeypatch):
        """extra='forbid' on InvokeRequest rejects the old audio_input_path
        field (or any other unknown key) with HTTP 422 -- POST /invoke is
        JSON-text-only now; audio goes exclusively through POST /invoke/audio.
        """
        server = _reloaded_server()

        client = TestClient(server.app)
        response = client.post(
            "/invoke",
            json={
                "input": "店の営業時間を教えてください",
                "session_id": "test-session-json",
                "audio_input_path": "/tmp/should-not-be-accepted.wav",
            },
        )

        assert response.status_code == 422
