"""Proof-of-Boundary structural check — CMN-C1-113."""

import os


def test_pb_scaffold_structure():
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    required = [
        "src/graph/graph.py",
        "src/nodes",
        "src/schemas/state.py",
        "src/api/server.py",
        "config/agent.yaml",
        "config/config.yaml",
    ]
    missing = [p for p in required if not os.path.exists(os.path.join(root, p))]
    assert not missing, f"Required scaffold paths missing: {missing}"
