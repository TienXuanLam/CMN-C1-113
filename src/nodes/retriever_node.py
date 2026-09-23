"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Never import from mediator/, api/, or other agents

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import VoiceFAQState

logger = logging.getLogger(__name__)

_DEFAULT_FAQ_KB = (
    {
        "text": "Support hours are Monday through Friday, 09:00 to 18:00 local business time, excluding public holidays.",
        "source": "default-faq/support-hours",
        "keywords": {"hours", "open", "close", "support", "営業時間", "受付時間"},
    },
    {
        "text": "For duplicate charges or refund requests, contact billing support with the order reference and payment date. The team will verify the transaction before confirming a refund.",
        "source": "default-faq/billing-refunds",
        "keywords": {"charged", "charge", "twice", "duplicate", "refund", "payment", "請求", "返金", "二重"},
    },
    {
        "text": "If you cannot access your account, use the password-reset option first. Contact account support if the reset link does not arrive or the account remains locked.",
        "source": "default-faq/account-access",
        "keywords": {"account", "login", "password", "access", "locked", "sign", "アカウント", "ログイン"},
    },
    {
        "text": "For a delayed order, check the tracking status and estimated delivery date. Contact order support if tracking has not updated or the delivery date has passed.",
        "source": "default-faq/order-delivery",
        "keywords": {"order", "delivery", "arrive", "tracking", "late", "届か", "配送", "注文"},
    },
)


class RetrieverNode(FunctionNode):
    """Vector similarity search over FAQ KB (Qdrant)."""

    required_trust_level = TrustLevel.ANONYMOUS

    def __init__(
        self,
        top_k: int = 5,
        score_threshold: float = 0.45,
        collection: str = "faq_kb",
        client_factory: Callable[[str], Any] | None = None,
    ):
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._collection = collection
        self._client_factory = client_factory

    def execute(self, state: VoiceFAQState) -> dict[str, Any]:
        emit_trace_event(
            event_type="retriever_start",
            payload={"session_id": state.get("session_id"), "transcript_chars": len(state.get("transcript") or "")},
            state=state,
        )

        if state.get("error_message"):
            return {}

        transcript = state.get("transcript")
        if transcript is None:
            return {"error_message": "RetrieverNode: transcript is None", "status": AgentStatus.ERROR.value}

        # Config from constructor (set by graph.register_nodes from config.yaml)
        # Allow per-request override via input_context, bounded -- Finding
        # (2026-08-19, High/Cost-DoS): top_k had no upper bound at all, so a
        # caller could request an arbitrarily large result set from Qdrant
        # (e.g. top_k=1000000), inflating query cost and the amount of
        # untrusted content later concatenated into the LLM prompt.
        _MAX_TOP_K = 20
        input_context = state.get("input_context", {})
        top_k: int = min(int(input_context.get("retriever_top_k", self._top_k)), _MAX_TOP_K)
        score_threshold: float = float(input_context.get("retriever_score_threshold", self._score_threshold))

        # F-05: validate collection name against allowlist to prevent cross-collection access
        requested_collection: str = input_context.get("qdrant_collection", self._collection)
        _ALLOWED_COLLECTIONS = {self._collection, "faq_kb"}
        if requested_collection not in _ALLOWED_COLLECTIONS:
            return {
                "error_message": f"RetrieverNode: collection '{requested_collection}' is not in the permitted allowlist",
                "status": AgentStatus.ERROR.value,
            }
        collection: str = requested_collection

        chunks: list[dict[str, Any]]
        if self._client_factory is not None:
            ctx = InvocationContext.from_state(state)
            qdrant_url: str = ctx.secrets.require("QDRANT_URL")
            try:
                client = self._client_factory(qdrant_url)
                results: list[Any] = client.query_points(
                    collection_name=collection,
                    query_text=transcript,
                    limit=top_k,
                ).points
            except AttributeError:
                return {
                    "error_message": "RetrieverNode: qdrant client does not support query_points",
                    "status": AgentStatus.ERROR.value,
                }
            except Exception as exc:  # noqa: BLE001
                return {
                    "error_message": f"RetrieverNode: KB query failed — {type(exc).__name__}",
                    "status": AgentStatus.ERROR.value,
                }
            chunks = [
                {
                    "text": str((hit.payload or {}).get("text", "")),
                    "score": float(hit.score),
                    "source": str((hit.payload or {}).get("source", "")),
                }
                for hit in results
            ]
        else:
            chunks = self._retrieve_default_faq(transcript, top_k)
        retrieval_score_max: float = max((float(c["score"]) for c in chunks), default=0.0)

        emit_trace_event(
            event_type="retriever_complete",
            payload={
                "session_id": state.get("session_id"),
                "result_count": len(chunks),
                "retrieval_score_max": retrieval_score_max,
                "low_confidence": retrieval_score_max < score_threshold,
            },
            state=state,
        )
        return {
            "retrieved_chunks": json.dumps(chunks, ensure_ascii=False),
            "retrieval_score_max": retrieval_score_max,
        }

    @staticmethod
    def _retrieve_default_faq(transcript: str, top_k: int) -> list[dict[str, Any]]:
        normalized = transcript.casefold()
        tokens = set(re.findall(r"[\w-]+", normalized, flags=re.UNICODE))
        ranked: list[dict[str, Any]] = []
        for entry in _DEFAULT_FAQ_KB:
            matches = sum(1 for keyword in entry["keywords"] if keyword in tokens or keyword in normalized)
            if matches:
                ranked.append(
                    {
                        "text": entry["text"],
                        "score": min(1.0, 0.55 + (0.15 * matches)),
                        "source": entry["source"],
                    }
                )
        ranked.sort(key=lambda item: float(item["score"]), reverse=True)
        return ranked[:top_k]
