"""AgentCore Platform v1.0"""

from typing import Any

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.stt_node import STTNode
from src.nodes.main_node import MainNode
from src.nodes.tts_node import TTSNode
from src.schemas.state import VoiceFAQState
from src.nodes.security_node import check_pii, check_output


class VoiceFAQGraph(AgentBaseGraph):
    """Fixed-pipeline graph for CMN-C1-113 VoiceActivatedFAQAgent.

    Slot mapping:
      pre_process  → STTNode      (already-transcribed text → transcript passthrough;
                                    audio bytes are transcribed in src/api/server.py
                                    before agent.invoke() is ever called — ADR-005)
      main         → MainNode     (RetrieverNode → GeneratorNode; uses retriever/llm config)
      post_process → TTSNode      (answer_text → audio; uses tts_chunk_max_chars from config)

    Runtime config keys (read from config/config.yaml via self.config):
      retriever_top_k, retriever_score_threshold, qdrant_collection,
      llm_model, tts_chunk_max_chars, timeout_s, max_retry — passed to nodes
      at register_nodes() time. self.config is populated by cli.py via
      framework.utils.config_loader.load_agent_config(), which reads
      config/config.yaml before run_agent_marketplace() constructs this
      graph.
    """

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    @property
    def name(self) -> str:
        return "cmn_c1_voice_faq_agent"

    @property
    def state_schema(self) -> type:
        return VoiceFAQState

    def register_nodes(self) -> None:
        super().register_nodes()
        self._nodes["pre_process"] = STTNode()
        self._nodes["main"] = MainNode(
            retriever_top_k=int(self.config.get("retriever_top_k", 5)),
            retriever_score_threshold=float(self.config.get("retriever_score_threshold", 0.45)),
            qdrant_collection=self.config.get("qdrant_collection", "faq_kb"),
            llm_model=self.config.get("llm_model", "gpt-4o-mini"),
            qdrant_client_factory=self.config.get("qdrant_client_factory"),
            openai_client_factory=self.config.get("openai_client_factory"),
            llm_client=self.config.get("llm"),
            timeout_s=float(self.config.get("timeout_s", 25.0)),
            max_retry=int(self.config.get("max_retry", 0)),
        )
        self._nodes["post_process"] = TTSNode(
            tts_chunk_max_chars=int(self.config.get("tts_chunk_max_chars", 200)),
        )

    def _enrich_output(self, output: dict[str, Any], state: dict[str, Any]) -> None:
        """Expose only the opaque audio handle, never the container file path."""
        audio_output_id = state.get("audio_output_id")
        if audio_output_id:
            output["audio_output_id"] = audio_output_id
            output["audio_duration_sec"] = state.get("audio_duration_sec")

    def _extra_security_gate_input(self, state: dict[str, Any]) -> dict[str, Any]:
        """S-2: PII scan on text_input and transcript. Short-circuits on first hit."""
        for field in ("user_input", "text_input", "transcript"):
            error = check_pii(state.get(field), field)
            if error:
                state["error_message"] = error
                state["status"] = AgentStatus.ERROR.value
                return state  # L-08: stop at first PII detection, don't overwrite
        return state

    def _extra_security_gate_output(self, result: dict[str, Any]) -> dict[str, Any]:
        """S-3: Credential + injection scan on answer_text."""
        error = check_output(result.get("answer_text"))
        if error:
            result["answer_text"] = None
            result["audio_output_path"] = None
            result["audio_output_id"] = None
            result["audio_duration_sec"] = None
            result["error_message"] = error
            result["status"] = AgentStatus.ERROR.value
        return result
