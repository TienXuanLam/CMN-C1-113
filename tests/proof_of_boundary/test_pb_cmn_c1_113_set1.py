"""Proof-of-Boundary tests — Set 1 (PB-1 to PB-6) — CMN-C1-113 VoiceActivatedFAQAgent.

Migrated to SDK pattern: FunctionNode, agent.invoke(), InvocationContext(caller_trust_level=).
"""

from __future__ import annotations

import ast
import json
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
        "session_id": "test-session-pb",
        "user_input": "",
        "text_input": "テスト入力",
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
# PB-1: emit_trace_event fires at each node boundary
# ---------------------------------------------------------------------------


class TestPB1TraceEventFires:
    """PB-1: Every node must call emit_trace_event at least once."""

    def test_stt_node_emits_trace_text_passthrough(self):
        node = STTNode()
        state = _base_state(text_input="こんにちは")
        events = []
        with patch("src.nodes.stt_node.emit_trace_event", side_effect=lambda **kw: events.append(kw)):
            with bound_secrets(_SECRETS):
                node.execute(state)
        assert len(events) >= 1, "STTNode must emit at least one trace event"

    def test_retriever_node_emits_trace(self):
        node = RetrieverNode()
        state = _base_state(transcript="店の営業時間は？")
        events = []
        with patch("src.nodes.retriever_node.emit_trace_event", side_effect=lambda **kw: events.append(kw)):
            with bound_secrets(_SECRETS):
                node.execute(state)
        assert len(events) >= 1, "RetrieverNode must emit at least one trace event"

    def test_generator_node_emits_trace(self):
        node = GeneratorNode()
        state = _base_state(transcript="店の営業時間は？", retrieved_chunks="[]", retrieval_score_max=0.0)
        events = []
        with patch("src.nodes.generator_node.emit_trace_event", side_effect=lambda **kw: events.append(kw)):
            with bound_secrets(_SECRETS):
                node.execute(state)
        assert len(events) >= 1, "GeneratorNode must emit at least one trace event"

    def test_tts_node_emits_trace_when_unavailable(self):
        node = TTSNode()
        state = _base_state(answer_text="ご質問ありがとうございます。")
        events = []
        with patch("src.nodes.tts_node.emit_trace_event", side_effect=lambda **kw: events.append(kw)):
            with bound_secrets(_SECRETS):
                node.execute(state)
        assert len(events) >= 1, "TTSNode must emit at least one trace event"

    def test_graph_slots_exist(self):
        """Graph must have pre_process, main, post_process slots after compile."""
        agent = VoiceFAQGraph()
        agent.compile()
        assert "pre_process" in agent._nodes
        assert "main" in agent._nodes
        assert "post_process" in agent._nodes


# ---------------------------------------------------------------------------
# PB-2: State serialisation — all values are primitives/JSON strings
# ---------------------------------------------------------------------------


class TestPB2StateSerialisation:
    """PB-2: After STTNode text-passthrough, all returned values must be primitives."""

    def test_stt_text_passthrough_state_all_primitives(self):
        node = STTNode()
        state = _base_state(text_input="営業時間を教えてください")
        with bound_secrets(_SECRETS):
            result = node.execute(state)
        for key, value in result.items():
            assert value is None or isinstance(
                value, (str, int, float, bool)
            ), f"State field '{key}' = {value!r} is not a primitive — Rule 1.1 violation"

    def test_retriever_error_state_all_primitives(self):
        node = RetrieverNode()
        state = _base_state(transcript="test query")
        with bound_secrets(_SECRETS):
            result = node.execute(state)
        for key, value in result.items():
            assert value is None or isinstance(
                value, (str, int, float, bool)
            ), f"State field '{key}' = {value!r} is not a primitive after RetrieverNode error"

    def test_retrieved_chunks_is_json_string(self):
        """PB-2: a real KB hit must reach state as a JSON *string*, not a list.

        The KB seam is now RetrieverNode(client_factory=...) — the node no
        longer imports QdrantClient at module scope nor exposes a
        _QDRANT_AVAILABLE flag, so the old patch() targets no longer exist.
        Injecting the client through the constructor exercises the same
        Qdrant code path (query_points(...).points) that production uses.
        """
        mock_hit = MagicMock()
        mock_hit.score = 0.85
        mock_hit.payload = {"text": "営業時間は9時から18時です。", "source": "faq_001"}
        mock_result = MagicMock()
        mock_result.points = [mock_hit]

        mock_client = MagicMock()
        mock_client.query_points.return_value = mock_result
        node = RetrieverNode(client_factory=lambda url: mock_client)

        state = _base_state(transcript="営業時間は？")
        with bound_secrets(_SECRETS):
            result = node.execute(state)

        assert mock_client.query_points.called, "the injected Qdrant client must actually be queried"
        assert isinstance(result["retrieved_chunks"], str), "retrieved_chunks must be str (JSON), not a list — Rule 1.1"
        parsed = json.loads(result["retrieved_chunks"])
        assert isinstance(parsed, list)
        assert parsed[0]["source"] == "faq_001"
        assert isinstance(result["retrieval_score_max"], float)
        assert result["retrieval_score_max"] == 0.85


# ---------------------------------------------------------------------------
# PB-3: L1 import isolation
# ---------------------------------------------------------------------------


class TestPB3ImportIsolation:
    """PB-3: AST scan of all src/ files — zero agenticstar imports."""

    def _scan_file(self, path: Path) -> list[str]:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "agenticstar" in alias.name:
                        violations.append(f"{path.name}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and "agenticstar" in node.module:
                    violations.append(f"{path.name}:{node.lineno}: from {node.module} import ...")
        return violations

    def test_no_agenticstar_imports_in_src(self):
        src_dir = ROOT / "src"
        all_violations = []
        for py_file in src_dir.rglob("*.py"):
            all_violations.extend(self._scan_file(py_file))
        assert all_violations == [], f"Import Isolation violation (Rule 2.1): {all_violations}"


# ---------------------------------------------------------------------------
# PB-4: S-1 trust enforcement
# ---------------------------------------------------------------------------


class TestPB4TrustEnforcement:
    """PB-4: Nodes must declare required_trust_level."""

    def test_stt_node_required_trust_level_is_verified_external(self):
        """Renamed from ..._is_internal: STTNode declares VERIFIED_EXTERNAL.

        STTNode is the pre_process entry node of a marketplace-invoked agent,
        so it must admit verified external callers; INTERNAL would reject
        every real caller at the S-1 gate. See src/nodes/stt_node.py.
        """
        node = STTNode()
        assert hasattr(node, "required_trust_level")
        assert node.required_trust_level == TrustLevel.VERIFIED_EXTERNAL

    def test_retriever_node_required_trust_level_is_anonymous(self):
        node = RetrieverNode()
        assert hasattr(node, "required_trust_level")
        assert node.required_trust_level == TrustLevel.ANONYMOUS

    def test_graph_required_trust_level_is_verified_external(self):
        """Renamed from ..._is_internal: the graph declares VERIFIED_EXTERNAL.

        PB-4's intent (every node/graph declares an explicit, consistent S-1
        level) is preserved — only the asserted constant is corrected to the
        level src/graph/graph.py actually declares.
        """
        graph = VoiceFAQGraph()
        assert graph.required_trust_level == TrustLevel.VERIFIED_EXTERNAL


# ---------------------------------------------------------------------------
# PB-5: Checkpoint safety
# ---------------------------------------------------------------------------


class TestPB5CheckpointSafety:
    """PB-5: State after STTNode contains no JWT tokens or Pydantic objects."""

    JWT_PATTERN = r"eyJ[a-zA-Z0-9._-]{10,}"

    def test_stt_passthrough_state_contains_no_jwt(self):
        import re

        node = STTNode()
        state = _base_state(text_input="テスト入力")
        with bound_secrets(_SECRETS):
            result = node.execute(state)
        for key, value in result.items():
            if isinstance(value, str):
                assert not re.search(
                    self.JWT_PATTERN, value
                ), f"JWT pattern found in state field '{key}' — checkpoint safety violation"

    def test_state_contains_no_pydantic_objects(self):
        node = STTNode()
        state = _base_state(text_input="安全テスト")
        with bound_secrets(_SECRETS):
            result = node.execute(state)
        for key, value in result.items():
            assert not hasattr(
                value, "__fields__"
            ), f"State field '{key}' is a Pydantic model — Rule 1.1 / PB-5 violation"

    def test_invocation_context_not_stored_in_state(self):
        node = STTNode()
        state = _base_state(text_input="テスト")
        with bound_secrets(_SECRETS):
            result = node.execute(state)
        from framework.schemas.invocation_context import InvocationContext as IC

        for key, value in result.items():
            assert not isinstance(value, IC), f"InvocationContext stored in state field '{key}' — Rule 1.3 violation"


# ---------------------------------------------------------------------------
# PB-6: Invoke execution order (SDK AgentBaseGraph)
# ---------------------------------------------------------------------------


class TestPB6InvokeExecutionOrder:
    """PB-6: SDK AgentBaseGraph pipeline order: initialize → pre_process → main → post_process → finalize."""

    def test_invoke_produces_node_history(self):
        """SDK node_history confirms pipeline execution order."""
        agent = VoiceFAQGraph()
        agent.compile()
        with bound_secrets(_SECRETS):
            result = agent.invoke(
                user_input="テスト",
                input_context={"text_input": "テスト"},
                ctx=_ctx(),
            )
        history = result.get("node_history", [])
        assert "STTNode" in history, "STTNode (pre_process) must appear in node_history"
        assert "MainNode" in history, "MainNode (main) must appear in node_history"
