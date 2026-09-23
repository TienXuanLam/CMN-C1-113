# CMN-C1-113 — Unit Tests: nodes


from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets.inmemory_provider import InMemoryProvider

_SECRETS = InMemoryProvider({"OPENAI_API_KEY": "test-key", "QDRANT_URL": "http://localhost:6333"})


def _base_state(**overrides) -> dict:
    state = {
        "user_input": "",
        # Unit tests call node.execute() directly — domain fields must be in top-level state.
        # (When running via agent.invoke(), STTNode gets them from input_context extracted
        # by the SDK pipeline; direct node calls bypass that extraction.)
        "text_input": "テストの質問です。",
        "input_context": {},
        "correlation_id": "test-corr",
        "session_id": "test-session",
        "thread_id": "test-thread",
        "trace_id": "",
        "caller_trust_level": "INTERNAL",
        "caller_id": "",
        "hitl_allowed": True,
        "node_history": [],
        "error_log": [],
        "error_code": None,
    }
    state.update(overrides)
    return state


class TestSTTNode:
    def test_text_passthrough(self):
        from src.nodes.stt_node import STTNode

        node = STTNode()
        with bound_secrets(_SECRETS):
            result = node.execute(_base_state())
        assert result.get("transcript") == "テストの質問です。"
        assert result.get("stt_language") == "text_passthrough"
        assert result.get("stt_confidence") == 1.0

    def test_text_passthrough_from_sdk_user_input(self):
        """SDK callers (POST /invoke, agent.invoke(user_input=...)) put the
        question in the top-level `user_input` state field, not
        `input_context` -- src/nodes/stt_node.py reads
        `state.get("user_input") or state.get("text_input")`, preferring
        user_input. Renamed from test_text_passthrough_from_sdk_input_context,
        which asserted an input_context["text_input"] path STTNode does not
        read.
        """
        from src.nodes.stt_node import STTNode

        state = _base_state(text_input=None, user_input="SDK input context question")

        with bound_secrets(_SECRETS):
            result = STTNode().execute(state)

        assert result.get("transcript") == "SDK input context question"
        assert result.get("status") is None

    def test_no_input_returns_input_guidance(self):
        """No audio and no text is a *user input* problem, not a system error.

        Renamed from test_no_input_returns_error: STTNode routes this case
        through build_input_guidance(), which deliberately returns
        status=success + input_validation_failed="true" + a Markdown
        formatted_output telling the caller what to send, and explicitly
        sets error_message=None. Asserting error_message is not None would
        now assert the opposite of the intended contract.
        """
        from src.nodes.stt_node import STTNode

        node = STTNode()
        with bound_secrets(_SECRETS):
            result = node.execute(_base_state(text_input=None))
        assert result.get("input_validation_failed") == "true"
        assert result.get("error_message") is None
        assert result.get("status") == "success"
        assert "The question is empty." in (result.get("formatted_output") or "")
        assert result.get("transcript") is None

    def test_stale_error_message_is_reset_not_propagated(self):
        """STTNode is the pipeline entry point, so it must CLEAR prior errors.

        Renamed from test_upstream_error_propagates. STTNode runs in the
        pre_process slot — no template node runs before it, so there is no
        upstream error for it to propagate. It instead emits a
        `transient_reset` block (error_message=None plus every downstream
        field) so a re-invoked session on the same thread does not inherit a
        stale error_message/transcript from the previous turn. The
        error_message early-return contract belongs to the *downstream*
        nodes (RetrieverNode/GeneratorNode/MainNode/TTSNode), which is
        covered by TestMainNode/TestTTSNode below and by PB-8.
        """
        from src.nodes.stt_node import STTNode

        node = STTNode()
        with bound_secrets(_SECRETS):
            result = node.execute(_base_state(error_message="stale error from a previous turn"))
        assert result["error_message"] is None
        assert result.get("transcript") == "テストの質問です。"

    def test_execute_method_contract(self):
        import inspect
        from src.nodes.stt_node import STTNode

        node = STTNode()
        sig = inspect.signature(node.execute)
        assert "state" in sig.parameters
        assert "_invoke_impl" not in STTNode.__dict__

    def test_required_trust_level(self):
        """STTNode accepts VERIFIED_EXTERNAL — it is the marketplace entry point.

        The agent is invoked by external marketplace callers, so gating the
        entry node at INTERNAL would reject every real caller. VERIFIED_EXTERNAL
        is the deliberate declared level in src/nodes/stt_node.py.
        """
        from src.nodes.stt_node import STTNode

        assert STTNode.required_trust_level == TrustLevel.VERIFIED_EXTERNAL


class TestMainNode:
    def test_emit_trace_event_on_upstream_error(self, monkeypatch):
        import src.nodes.main_node as main_module

        events = []
        monkeypatch.setattr(main_module, "emit_trace_event", lambda **kwargs: events.append(kwargs))

        result = main_module.MainNode().execute(_base_state(error_message="upstream error"))

        assert result == {}
        assert [event["event_type"] for event in events] == ["main_pipeline_started"]
        assert events[0]["payload"] == {"has_upstream_error": True}

    def test_execute_contract(self):
        import inspect
        from src.nodes.main_node import MainNode

        node = MainNode()
        sig = inspect.signature(node.execute)
        assert "state" in sig.parameters
        assert "_invoke_impl" not in MainNode.__dict__

    def test_required_trust_level(self):
        """MainNode matches the graph's VERIFIED_EXTERNAL level.

        MainNode sits in the main slot of a graph declared VERIFIED_EXTERNAL;
        an INTERNAL requirement here would make the S-1 gate reject callers
        the graph itself admits. See src/nodes/main_node.py.
        """
        from src.nodes.main_node import MainNode

        assert MainNode.required_trust_level == TrustLevel.VERIFIED_EXTERNAL


class TestTTSNode:
    def test_missing_answer_returns_error(self):
        from src.nodes.tts_node import TTSNode

        node = TTSNode()
        with bound_secrets(_SECRETS):
            result = node.execute(_base_state(answer_text=None))
        assert result.get("error_message") is not None

    def test_upstream_error_propagates(self):
        from src.nodes.tts_node import TTSNode

        node = TTSNode()
        with bound_secrets(_SECRETS):
            result = node.execute(_base_state(error_message="upstream error"))
        assert result == {}
