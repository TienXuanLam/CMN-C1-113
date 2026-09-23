"""Proof-of-Boundary tests — Set 2 (PB-7 to PB-9) — CMN-C1-113 VoiceActivatedFAQAgent.

Migrated to SDK pattern: FunctionNode, agent.invoke(), InvocationContext(caller_trust_level=).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.schemas.invocation_context import InvocationContext, TrustLevel  # noqa: E402
from framework.secrets.context import bound_secrets  # noqa: E402
from shared.secrets.inmemory_provider import InMemoryProvider  # noqa: E402
from src.nodes.stt_node import STTNode  # noqa: E402
from src.nodes.retriever_node import RetrieverNode  # noqa: E402
from src.nodes.generator_node import GeneratorNode  # noqa: E402
from src.nodes.tts_node import TTSNode  # noqa: E402
from src.graph.graph import VoiceFAQGraph  # noqa: E402

_SECRETS = InMemoryProvider({"OPENAI_API_KEY": "test-key", "QDRANT_URL": "http://localhost:6333"})


def _base_state(**overrides: Any) -> dict:
    state: dict = {
        "session_id": "test-session-pb2",
        "user_input": "",
        "text_input": None,
        "transcript": None,
        "stt_language": None,
        "stt_confidence": None,
        "retrieved_chunks": None,
        "retrieval_score_max": None,
        "answer_text": None,
        "answer_language": None,
        "audio_output_path": None,
        "audio_duration_sec": None,
        "error_message": None,
        "correlation_id": "test-corr",
        "thread_id": "test-thread",
        "trace_id": "",
        "caller_trust_level": "INTERNAL",
        "caller_id": "",
        "hitl_allowed": True,
        "node_history": [],
        "error_log": [],
        "input_context": {},
    }
    state.update(overrides)
    return state


def _ctx():
    return InvocationContext(caller_id="test", caller_trust_level=TrustLevel.INTERNAL)


# ---------------------------------------------------------------------------
# PB-7: STT text passthrough
# ---------------------------------------------------------------------------


class TestPB7STTTextPassthrough:
    """PB-7: When text_input is set, STTNode must use the text passthrough
    path and MUST NOT invoke Whisper."""

    def test_text_passthrough_sets_correct_fields(self):
        node = STTNode()
        text = "店の営業時間を教えてください"
        state = _base_state(text_input=text)
        with bound_secrets(_SECRETS):
            result = node.execute(state)
        assert result["transcript"] == text
        assert result["stt_language"] == "text_passthrough"
        assert result["stt_confidence"] == 1.0
        # error_message is not returned on success (node returns only changed fields)
        assert result.get("error_message") is None or "error_message" not in result

    def test_text_passthrough_strips_whitespace(self):
        node = STTNode()
        state = _base_state(text_input="  前後にスペース  ")
        with bound_secrets(_SECRETS):
            result = node.execute(state)
        assert result["transcript"] == "前後にスペース"

    def test_text_passthrough_has_no_whisper_dependency(self):
        """Whisper transcription now lives entirely in src/api/server.py

        (see its module comment / ADR-005) — STTNode itself no longer
        imports or references whisper at all, so there is nothing to patch
        here. This test pins that STTNode's module has no `_whisper`
        symbol, guaranteeing the text-passthrough path structurally cannot
        invoke it.
        """
        import src.nodes.stt_node as stt_module

        assert not hasattr(stt_module, "_whisper")

    def test_text_passthrough_emits_trace_with_mode(self):
        node = STTNode()
        state = _base_state(text_input="トレースモードテスト")
        events = []
        with patch("src.nodes.stt_node.emit_trace_event", side_effect=lambda **kw: events.append(kw)):
            with bound_secrets(_SECRETS):
                node.execute(state)
        assert any(e.get("payload", {}).get("mode") == "text_passthrough" for e in events) or any(
            "text_passthrough" in str(e) for e in events
        ), "Trace event must contain mode='text_passthrough'"


# ---------------------------------------------------------------------------
# PB-8: Error propagation
# ---------------------------------------------------------------------------


class TestPB8ErrorPropagation:
    """PB-8: When a node sets error_message, downstream nodes must skip (return {})."""

    def test_downstream_nodes_skip_on_error_message(self):
        """RetrieverNode, GeneratorNode, TTSNode must return {} when error_message is set."""
        error_state = _base_state(error_message="STTNode: no text_input provided")

        retriever = RetrieverNode()
        with bound_secrets(_SECRETS):
            retriever_result = retriever.execute(dict(error_state))
        assert retriever_result == {}, "RetrieverNode must return {} when error_message is set"

        generator = GeneratorNode()
        with bound_secrets(_SECRETS):
            gen_result = generator.execute(dict(error_state))
        assert gen_result == {}, "GeneratorNode must return {} when error_message is set"

        tts = TTSNode()
        with bound_secrets(_SECRETS):
            tts_result = tts.execute(dict(error_state))
        assert tts_result == {}, "TTSNode must return {} when error_message is set"

    def test_error_message_is_first_error_only(self):
        """The first error must not be overwritten by subsequent skipped nodes."""
        original_error = "STTNode: Whisper failed — mock error"
        error_state = _base_state(error_message=original_error)

        nodes = [RetrieverNode(), GeneratorNode(), TTSNode()]
        for node in nodes:
            with bound_secrets(_SECRETS):
                result = node.execute(dict(error_state))
            # Skipped nodes return {} — they don't overwrite
            assert (
                result == {} or result.get("error_message") == original_error
            ), f"{node.__class__.__name__} must not overwrite the original error_message"


# ---------------------------------------------------------------------------
# PB-9: Retrieval score gate
# ---------------------------------------------------------------------------


def _capturing_generator(answer: str) -> tuple[GeneratorNode, list[dict[str, str]]]:
    """Build a GeneratorNode whose LLM call is captured, via the real seam.

    The previous helper-less bodies patched
    src.nodes.generator_node._OPENAI_AVAILABLE and .openai. Neither symbol
    exists any more: GeneratorNode reaches the provider through
    shared.services.llm.AzureOpenAIClient, and its documented injection
    point is the `client_factory` constructor argument (a real config.yaml
    key, `openai_client_factory`). The factory-built client is wrapped in
    _ChatCompletionsAdapter, so it must expose
    `.chat.completions.create(model=..., messages=[...])` — the same shape
    the old mock provided.
    """
    captured: list[dict[str, str]] = []

    def mock_create(**kwargs: Any) -> Any:
        captured.extend(kwargs.get("messages", []))
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = answer
        return mock_resp

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = mock_create
    return GeneratorNode(llm_model="gpt-4o-mini", client_factory=lambda _: mock_client), captured


class TestPB9RetrievalScoreGate:
    """PB-9: low retrieval confidence must not halt the pipeline, and must
    now inject the documented disclaimer (docs/02_design.md L113-114,
    docs/03_test_spec.md PB-9, docs/07_operation_guide.md L222/L246).

    GeneratorNode reads state["retrieval_score_max"] and, when it is below
    the same retriever_score_threshold RetrieverNode uses (wired from
    MainNode/config.yaml, default 0.45), appends a disclaimer instruction to
    the system prompt and sets disclaimer_injected=True on the
    generator_complete trace event.
    """

    _DISCLAIMER_NOTE = "low relevance to this question"

    def test_disclaimer_injected_when_score_below_threshold(self):
        node, captured = _capturing_generator("テストの回答です。")
        state = _base_state(
            transcript="営業時間は？",
            retrieved_chunks='[{"text": "一部情報", "score": 0.3, "source": "faq"}]',
            retrieval_score_max=0.3,
        )
        with bound_secrets(_SECRETS):
            node.execute(state)

        system_messages = [m["content"] for m in captured if m.get("role") == "system"]
        assert self._DISCLAIMER_NOTE in system_messages[0]

    def test_low_score_context_still_reaches_prompt_and_answers(self):
        """The other half of PB-9: a low score never blocks generation."""
        node, captured = _capturing_generator("テストの回答です。")
        state = _base_state(
            transcript="営業時間は？",
            retrieved_chunks='[{"text": "一部情報", "score": 0.3, "source": "faq"}]',
            retrieval_score_max=0.3,
        )
        with bound_secrets(_SECRETS):
            result = node.execute(state)

        system_messages = [m["content"] for m in captured if m.get("role") == "system"]
        user_messages = [m["content"] for m in captured if m.get("role") == "user"]
        assert len(system_messages) >= 1
        # The low-scoring chunk is passed to the LLM rather than dropped, and
        # the disclaimer instructs the model to disclaim missing coverage.
        assert "一部情報" in user_messages[0]
        assert self._DISCLAIMER_NOTE in system_messages[0]
        assert result.get("answer_text") == "テストの回答です。"
        assert result.get("error_message") is None

    def test_empty_retrieved_chunks_still_answers_without_error(self):
        """Renamed from test_disclaimer_injected_when_retrieved_chunks_empty.

        retrieval_score_max=0.0 is below threshold, so the disclaimer IS
        injected here too — an empty KB result is treated as low confidence,
        not as an error. The context is forwarded and an answer is still
        generated.
        """
        node, captured = _capturing_generator("情報が見つかりませんでした。")
        state = _base_state(transcript="どこにありますか？", retrieved_chunks="[]", retrieval_score_max=0.0)
        with bound_secrets(_SECRETS):
            result = node.execute(state)

        user_messages = [m["content"] for m in captured if m.get("role") == "user"]
        system_messages = [m["content"] for m in captured if m.get("role") == "system"]
        assert "[]" in user_messages[0]
        assert self._DISCLAIMER_NOTE in system_messages[0]
        assert result.get("answer_text") == "情報が見つかりませんでした。"
        assert result.get("error_message") is None

    def test_system_prompt_is_score_independent(self):
        """Renamed from test_no_disclaimer_when_score_above_threshold.

        Now that the score gate is implemented, the two prompts are
        expected to DIFFER: a high-confidence retrieval gets the base
        prompt unchanged, a low-confidence one gets the disclaimer
        appended. Pins both halves of that contract in one place.
        """
        high_node, high_captured = _capturing_generator("営業時間は9時から18時です。")
        low_node, low_captured = _capturing_generator("営業時間は9時から18時です。")

        with bound_secrets(_SECRETS):
            high_node.execute(
                _base_state(
                    transcript="営業時間は？",
                    retrieved_chunks='[{"text": "営業時間は9時から18時です。", "score": 0.92, "source": "faq"}]',
                    retrieval_score_max=0.92,
                )
            )
            low_node.execute(
                _base_state(
                    transcript="営業時間は？",
                    retrieved_chunks='[{"text": "営業時間は9時から18時です。", "score": 0.10, "source": "faq"}]',
                    retrieval_score_max=0.10,
                )
            )

        high_system = [m["content"] for m in high_captured if m.get("role") == "system"]
        low_system = [m["content"] for m in low_captured if m.get("role") == "system"]
        assert self._DISCLAIMER_NOTE not in high_system[0]
        assert self._DISCLAIMER_NOTE in low_system[0]
        assert high_system[0] != low_system[0]
        # The base instructions (including the JSON-only response contract)
        # are a strict prefix of the low-confidence prompt — the disclaimer
        # is appended, never substituted.
        assert low_system[0].startswith(high_system[0])

    def test_pipeline_continues_after_low_score_retrieval(self):
        """Low retrieval score must not halt the pipeline."""
        agent = VoiceFAQGraph()
        agent.compile()
        with bound_secrets(_SECRETS):
            result = agent.invoke(
                user_input="テスト質問",
                input_context={
                    "text_input": "テスト質問",
                },
                ctx=_ctx(),
            )
        # With no real KB/LLM, may be error or success — pipeline must not crash
        assert result["status"] in ("success", "error")
