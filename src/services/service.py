"""AgentCore Platform v1.0"""

from __future__ import annotations
from typing import Any


class Service:
    """Domain service placeholder for CMN-C1-113."""

    async def fetch(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError("Implement fetch() for Service")
