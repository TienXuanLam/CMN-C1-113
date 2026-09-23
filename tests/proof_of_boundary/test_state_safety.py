# PB: State Safety — CMN-C1-113

import ast
import importlib
import os
import re
import pytest

_CREDENTIAL_RE = re.compile(r"(?:^|_)(api_key|secret|password|credential|jwt|bearer|token)(?:_|$)", re.IGNORECASE)
_PROHIBITED = ["BaseModel", "InvocationContext"]


def _state_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "schemas", "state.py"))


def _parse():
    with open(_state_path()) as f:
        return ast.parse(f.read())


def _get_class(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "VoiceFAQState":
            return node
    pytest.fail("VoiceFAQState not found in src/schemas/state.py")


def _fields(cls):
    return [
        (item.target.id, item)
        for item in cls.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    ]


class TestNoCredentialFields:
    def test_no_credential_names(self):
        cls = _get_class(_parse())
        violations = [f"  {n}" for n, _ in _fields(cls) if _CREDENTIAL_RE.search(n)]
        assert not violations, "Credential-like field names:\n" + "\n".join(violations)


class TestMsgpackSafe:
    def test_no_prohibited_types(self):
        cls = _get_class(_parse())
        violations = []
        for name, item in _fields(cls):
            if item.annotation:
                names = {n.id for n in ast.walk(item.annotation) if isinstance(n, ast.Name)}
                for p in _PROHIBITED:
                    if p in names:
                        violations.append(f"  {name} uses {p}")
        assert not violations, "Prohibited types:\n" + "\n".join(violations)


class TestRequiredFields:
    _REQUIRED = {
        "text_input",
        "transcript",
        "stt_language",
        "stt_confidence",
        "retrieved_chunks",
        "retrieval_score_max",
        "answer_text",
        "answer_language",
        "audio_output_path",
        "audio_duration_sec",
        "error_message",
    }

    def test_all_required_fields(self):
        cls = _get_class(_parse())
        declared = {n for n, _ in _fields(cls)}
        missing = self._REQUIRED - declared
        assert not missing, f"Missing fields: {sorted(missing)}"


class TestRuntimeSafety:
    def test_state_is_not_pydantic(self):
        mod = importlib.import_module("src.schemas.state")
        cls = getattr(mod, "VoiceFAQState")
        assert hasattr(cls, "__annotations__")
        try:
            from pydantic import BaseModel

            assert not issubclass(cls, BaseModel)
        except ImportError:
            pass

    def test_inherits_agent_state(self):
        mod = importlib.import_module("src.schemas.state")
        cls = getattr(mod, "VoiceFAQState")
        from framework.schemas.agent_state import AgentState

        missing = set(AgentState.__annotations__) - set(cls.__annotations__)
        assert not missing, f"Missing AgentState fields: {sorted(missing)}"
