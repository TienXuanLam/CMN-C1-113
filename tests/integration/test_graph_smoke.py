# CMN-C1-113 — Integration smoke test

from framework.schemas.invocation_context import InvocationContext, TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets.inmemory_provider import InMemoryProvider
from src.graph.graph import VoiceFAQGraph

_SECRETS = InMemoryProvider({"OPENAI_API_KEY": "test-key", "QDRANT_URL": "http://localhost:6333"})
_VALID_INPUT = "テストの質問です。"


def _ctx():
    return InvocationContext(caller_id="test", caller_trust_level=TrustLevel.INTERNAL)


def _build_agent():
    agent = VoiceFAQGraph()
    agent.compile()
    return agent


def test_graph_compiles():
    agent = _build_agent()
    assert agent._compiled is not None


def test_text_passthrough_reaches_stt():
    """STTNode should run and reach the whole pipeline even without LLM/KB.

    The old body monkeypatched retriever_node._QDRANT_AVAILABLE,
    generator_node._OPENAI_AVAILABLE and openai.OpenAI. None of those
    symbols exist any more: RetrieverNode falls back to its in-process
    _DEFAULT_FAQ_KB when no client_factory is injected, and GeneratorNode
    goes through shared.services.llm.AzureOpenAIClient. With no real
    provider reachable, GeneratorNode's own except-branch converts the
    failure into a build_input_guidance() response, so the pipeline still
    completes end to end — which is exactly what this smoke test asserts.
    """
    agent = _build_agent()
    with bound_secrets(_SECRETS):
        result = agent.invoke(
            user_input=_VALID_INPUT,
            input_context={"text_input": _VALID_INPUT},
            ctx=_ctx(),
        )

    # Either success (guidance / answer) or error — but every slot ran.
    assert result["status"] in ("success", "error")
    assert [n for n in result["node_history"] if "STTNode" in n], "STTNode must have executed"
    assert "MainNode" in result["node_history"]


def test_empty_input_returns_input_guidance():
    """Renamed from test_empty_input_returns_error.

    An empty question is a user-input problem, not a system failure:
    STTNode routes it through build_input_guidance(), which returns
    status=success plus a Markdown prompt telling the caller what to send.
    The original intent (empty input must not produce a normal answer and
    must be reported back to the caller) is preserved by asserting the
    guidance output instead of an error status.
    """
    agent = _build_agent()
    with bound_secrets(_SECRETS):
        result = agent.invoke(
            user_input="",
            input_context={"text_input": None},
            ctx=_ctx(),
        )
    assert result["status"] == "success"
    assert "Customer FAQ question required" in result["output"]
    assert "The question is empty." in result["output"]
