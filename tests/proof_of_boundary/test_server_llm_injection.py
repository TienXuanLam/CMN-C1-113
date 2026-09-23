# src/api/server.py resolves AZURE_OPENAI_* from os.environ first, falling
# back to the configured SecretProvider (see server.py's own finding
# comment) so a missing key degrades _secrets_provider to an empty
# InMemoryProvider instead of crashing the process at import — the
# standalone deployment path provisions no key at boot, only per-invocation
# inside GeneratorNode._build_llm() via ctx.secrets.require(). Unlike the
# scaffold's server.py, CMN-C1-113 does NOT construct config["llm"] at
# import time — MainNode/GeneratorNode resolve the client fresh on every
# invocation, so there is no server._llm / server.agent._nodes["main"]._llm
# identity to assert here (scaffold's TestServerConstructsLlmWithKey has no
# equivalent in this file).

import importlib


class TestServerBootsWithoutAzureKeys:
    def test_server_imports_and_app_constructs_with_no_key(self, monkeypatch):
        for key in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"):
            monkeypatch.delenv(key, raising=False)

        import src.api.server as server

        importlib.reload(server)

        assert server.app is not None
        assert server.agent is not None
        assert server._secrets_provider.get("AZURE_OPENAI_API_KEY") is None


class TestServerConstructsLlmWithKey:
    def test_llm_is_constructed_and_reaches_main_node(self, monkeypatch):
        # Finding: server.py reads os.environ first (see its own module
        # comment) so environment variables are the direct way to exercise
        # this path -- unlike the scaffold's equivalent test, monkeypatching
        # shared.secrets.factory alone would not be observed because
        # os.environ.get(key) short-circuits before the configured provider
        # is ever consulted.
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-test-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.services.ai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "dummy-test-deployment")

        import src.api.server as server

        importlib.reload(server)

        from framework.secrets.context import bound_secrets
        from shared.services.llm.azure_openai_client import AzureOpenAIClient

        # server.agent.compile() normally runs inside the FastAPI lifespan
        # context manager (_lifespan), not at module import time -- populate
        # _nodes explicitly here since this test only imports the module.
        server.agent.compile()

        with bound_secrets(server._secrets_provider):
            state = {
                "correlation_id": "test-corr",
                "session_id": "test-session",
                "thread_id": "test-thread",
                "trace_id": "",
            }
            llm = server.agent._nodes["main"]._generator._build_llm(state)
        assert isinstance(llm, AzureOpenAIClient)


class TestStandaloneTrustPromotion:
    def test_external_bearer_never_promotes_to_internal(self):
        import src.api.server as server
        from framework.schemas.trust_level import TrustLevel

        assert (
            server._resolve_standalone_trust(TrustLevel.ANONYMOUS, "Bearer external", "external", "runner")
            is TrustLevel.VERIFIED_EXTERNAL
        )

    def test_runner_bearer_promotes_to_internal(self):
        import src.api.server as server
        from framework.schemas.trust_level import TrustLevel

        assert (
            server._resolve_standalone_trust(TrustLevel.ANONYMOUS, "Bearer runner", "external", "runner")
            is TrustLevel.INTERNAL
        )

    def test_wrong_or_missing_bearer_is_rejected_when_auth_is_enabled(self):
        import pytest
        import src.api.server as server
        from fastapi import HTTPException
        from framework.schemas.trust_level import TrustLevel

        for authorization in ("", "Bearer wrong"):
            with pytest.raises(HTTPException) as exc:
                server._resolve_standalone_trust(TrustLevel.ANONYMOUS, authorization, "external", "runner")
            assert exc.value.status_code == 401
