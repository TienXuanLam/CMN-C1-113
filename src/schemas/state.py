"""AgentCore Platform v1.0"""

# ADR-005: State must be a flat TypedDict — never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption. Extend AgentState with agent-specific
# fields only. Do NOT add credentials, secrets, or Pydantic models.

from typing import Optional

from framework.schemas.agent_state import AgentState


class VoiceFAQState(AgentState):
    """Agent state for CMN-C1-113 VoiceActivatedFAQAgent.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.

    Pipeline: STTNode → RetrieverNode → GeneratorNode → TTSNode
    """

    # ---------- Input ----------
    text_input: Optional[str]
    input_mode: Optional[str]
    audio_response_requested: Optional[bool]
    input_validation_failed: Optional[str]

    # ---------- STT output ----------
    transcript: Optional[str]
    stt_language: Optional[str]
    stt_confidence: Optional[float]

    # ---------- Retriever output ----------
    retrieved_chunks: Optional[str]
    retrieval_score_max: Optional[float]

    # ---------- Generator output ----------
    answer_text: Optional[str]
    answer_language: Optional[str]

    # ---------- TTS output ----------
    audio_output_path: Optional[str]
    audio_output_id: Optional[str]
    audio_duration_sec: Optional[float]

    # ---------- Output ----------
    formatted_output: Optional[str]

    # ---------- Error propagation ----------
    error_message: Optional[str]
