"""AgentCore Platform v1.0"""

# Composite main slot node: RetrieverNode → GeneratorNode sequentially.
# STTNode is pre_process; TTSNode is post_process.
#
# L-02 note: state passed into execute() is the FULL graph state from AgentBaseGraph,
# not a partial delta. transcript (set by STTNode in pre_process slot) is present
# in state when MainNode.execute() is called. {**state, **accumulated} merges
# accumulated delta on top of full state for each inner node call.

from collections.abc import Callable
from typing import Any, cast

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.services.events import emitter
from shared.services.events.types import EventType
from shared.services.llm.base_llm import BaseLLM
from shared.utils.audit_logger import emit_trace_event

from src.nodes.retriever_node import RetrieverNode
from src.nodes.generator_node import GeneratorNode
from src.schemas.state import VoiceFAQState


class MainNode(FunctionNode):
    """Compose RetrieverNode → GeneratorNode in the main SDK slot."""

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        retriever_top_k: int = 5,
        retriever_score_threshold: float = 0.45,
        qdrant_collection: str = "faq_kb",
        llm_model: str = "gpt-4o-mini",
        qdrant_client_factory: Callable[[str], Any] | None = None,
        openai_client_factory: Callable[[str], Any] | None = None,
        llm_client: BaseLLM | None = None,
        timeout_s: float = 25.0,
        max_retry: int = 0,
    ):
        self._retriever = RetrieverNode(
            top_k=retriever_top_k,
            score_threshold=retriever_score_threshold,
            collection=qdrant_collection,
            client_factory=qdrant_client_factory,
        )
        self._generator = GeneratorNode(
            llm_model=llm_model,
            client_factory=openai_client_factory,
            llm_client=llm_client,
            timeout_s=timeout_s,
            max_retry=max_retry,
            retrieval_score_threshold=retriever_score_threshold,
        )

    def execute(self, state: VoiceFAQState) -> dict[str, Any]:
        # S-4 domain audit for the composite slot. Inner nodes deliberately use
        # execute() directly so their framework lifecycle is not duplicated;
        # MainNode must therefore emit its own non-sensitive domain event.
        emit_trace_event(
            event_type="main_pipeline_started",
            payload={"has_upstream_error": bool(state.get("error_message"))},
            state=state,
        )

        if state.get("input_validation_failed") == "true" or state.get("error_message"):
            return {}

        # L-04: accumulated starts empty; {**state, **accumulated} is {**state} on first call.
        # On second call it merges retriever results on top of full state so GeneratorNode
        # receives both transcript (from state) and retrieved_chunks (from accumulated).
        accumulated: dict[str, Any] = {}

        # Finding (2026-08-19, CI regression): calling self._retriever(...) /
        # self._generator(...) invokes BaseNode.__call__(), which re-runs the
        # full S-1/S-4 invoke lifecycle (its own node_start/node_complete
        # events and a second S-1 trust check) for each inner node --
        # BaseNode.__call__'s own docstring says "templates must not
        # duplicate them". Confirmed live via
        # tests/proof_of_boundary/test_pb_invoke_order.py: MainNode's single
        # outer invocation produced two node_start/node_complete pairs
        # instead of one. Call .execute() directly -- MainNode's own
        # __call__ (inherited, unmodified) already provides the one
        # S-1/S-4/S-2/S-3 cycle for this composite as a whole.
        # cast: {**state, **accumulated} is structurally a VoiceFAQState (a
        # full-state dict overlaid with an in-progress delta of the same
        # shape) -- mypy can't see that through dict merge syntax.
        emitter().emit_event(
            event_type=EventType.PROGRESS_UPDATE,
            message="Retrieving relevant FAQ information.",
            metadata={"stage": "faq_retrieval"},
        )
        result = self._retriever.execute(cast(VoiceFAQState, {**state, **accumulated}))
        accumulated.update(result)
        if accumulated.get("error_message"):
            return accumulated

        emitter().emit_event(
            event_type=EventType.PROGRESS_UPDATE,
            message="Generating an answer with Azure OpenAI.",
            metadata={"stage": "answer_generation"},
        )
        result = self._generator.execute(cast(VoiceFAQState, {**state, **accumulated}))
        accumulated.update(result)

        if "status" not in accumulated:
            accumulated["status"] = AgentStatus.SUCCESS.value

        return accumulated
