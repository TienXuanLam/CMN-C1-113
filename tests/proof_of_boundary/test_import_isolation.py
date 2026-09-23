# PB-4: Import Isolation — CMN-C1-113

import ast
import os

_L0_PROHIBITED = ["agenticstar", "agenticstar_agentcore"]


def _scan_imports(filepath):
    with open(filepath) as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return []
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for p in _L0_PROHIBITED:
                    if alias.name == p or alias.name.startswith(f"{p}."):
                        violations.append(f"{filepath}:{node.lineno} — import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            for p in _L0_PROHIBITED:
                if node.module == p or node.module.startswith(f"{p}."):
                    violations.append(f"{filepath}:{node.lineno} — from {node.module} import ...")
    return violations


def _find_py(directory, exclude=None):
    exclude = exclude or set()
    files = []
    for root, dirs, fs in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in exclude]
        files.extend(os.path.join(root, f) for f in fs if f.endswith(".py"))
    return files


_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class TestImportIsolation:
    def test_no_l0_imports_in_src(self):
        src_dir = os.path.join(_root, "src")
        violations = []
        for f in _find_py(src_dir):
            violations.extend(_scan_imports(f))
        assert violations == [], "L0 violations:\n" + "\n".join(violations)

    def test_src_uses_framework_imports(self):
        src_dir = os.path.join(_root, "src")
        found = False
        for f in _find_py(src_dir):
            with open(f) as fh:
                try:
                    tree = ast.parse(fh.read())
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("framework."):
                    found = True
                    break
            if found:
                break
        assert found, "No framework.* imports found in src/"
