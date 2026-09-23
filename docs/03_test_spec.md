# CMN-C1-113 — Test Specification

## Metadata

| Field | Value |
|-------|-------|
| Template ID | CMN-C1-113 |
| Agent Name | VoiceActivatedFAQAgent |
| SDK | `agenticstar-agentcore==1.0.0rc1` |
| Stage | stg-complete |

---

## 1. Test Environment

- **Runner:** `pytest` via the shared CI `run-tests` job (wheel install)
- **Framework:** SDK wheel — no legacy pre-wheel provisioning
- **Secrets:** `InMemoryProvider` + `bound_secrets` for all tests requiring secrets
- **LLM/STT/TTS:** mocked via `unittest.mock.patch` or availability guards

---

## 2. Unit Tests — Node Level (`tests/unit/test_main_node.py`)

### STTNode

| TC-ID | Test | Expected |
|-------|------|----------|
| TC-STT-01 | `text_input` set | `transcript=text_input`, `stt_language="text_passthrough"`, `stt_confidence=1.0` |
| TC-STT-02 | Both None | `error_message` set, returns error dict |
| TC-STT-03 | Upstream `error_message` set | Returns `{}` immediately |
| TC-STT-04 | `execute()` method contract | `"state"` in signature, no `_invoke_impl` |
| TC-STT-05 | `required_trust_level` | `TrustLevel.INTERNAL` |

### MainNode

| TC-ID | Test | Expected |
|-------|------|----------|
| TC-MAIN-01 | `execute()` method contract | `"state"` in signature, no `_invoke_impl` |
| TC-MAIN-02 | `required_trust_level` | `TrustLevel.INTERNAL` |

### TTSNode

| TC-ID | Test | Expected |
|-------|------|----------|
| TC-TTS-01 | `answer_text=None` | `error_message` set |
| TC-TTS-02 | Upstream `error_message` set | Returns `{}` immediately |

---

## 3. Proof-of-Boundary Tests

### Set 1 (PB-1 to PB-6) — `tests/proof_of_boundary/test_pb_cmn_c1_113_set1.py`

| PB-ID | Boundary | Test | Expected |
|-------|----------|------|----------|
| PB-1 | `emit_trace_event` fires | Patch module-level `emit_trace_event` in each node | ≥1 event captured per node |
| PB-2 | State serialisation | All values returned by nodes are primitives/None | No Pydantic/dataclass/arbitrary objects |
| PB-3 | L1 import isolation | AST scan of `src/` | 0 `agenticstar` imports |
| PB-4 | S-1 trust declaration | `node.required_trust_level` attribute | `TrustLevel.INTERNAL` or `TrustLevel.ANONYMOUS` |
| PB-5 | Checkpoint safety | No JWT/Pydantic in STTNode result | Clean state |
| PB-6 | Invoke order | `agent.invoke()` → `node_history` contains expected nodes | STTNode, MainNode in history |

### Set 2 (PB-7 to PB-9) — `tests/proof_of_boundary/test_pb_cmn_c1_113_set2.py`

| PB-ID | Boundary | Test | Expected |
|-------|----------|------|----------|
| PB-7 | STT text passthrough | `text_input` only, no Whisper | `transcript=text_input`, `mode="text_passthrough"` in trace |
| PB-8 | Error propagation | Downstream nodes when `error_message` set | All return `{}` (SDK: first error preserved in state) |
| PB-9 | Retrieval score gate | `retrieval_score_max < 0.45` → disclaimer | Disclaimer in system prompt; pipeline continues |

---

## 4. Integration Tests (`tests/integration/test_graph_smoke.py`)

| TC-ID | Test | Expected |
|-------|------|----------|
| TC-INT-01 | `agent.compile()` | `agent._compiled is not None` |
| TC-INT-02 | Text input → full pipeline | `result["status"] in ("success", "error")` (no crash) |
| TC-INT-03 | Empty input | `result["status"] == "error"` |

---

## 5. S-2 Input Gate Tests

| TC-ID | Threat | Pattern | Expected |
|-------|--------|---------|----------|
| TC-S2-01 | PII | Credit card in `text_input` | `error_message` set, `status=error` |
| TC-S2-02 | PII | My Number (12-digit) | `error_message` set |
| TC-S2-03 | PII | Email address | `error_message` set |
| TC-S2-04 | PII | JP phone number | `error_message` set |
| TC-S2-05 | PII | JP postal code in `transcript` | `error_message` set |
| TC-S2-06 | Clean | No PII patterns | Gate passes |

---

## 6. S-3 Output Gate Tests

| TC-ID | Threat | Pattern | Expected |
|-------|--------|---------|----------|
| TC-S3-01 | Credential | OpenAI key (`sk-`) in `answer_text` | `answer_text` cleared, `error_message` set |
| TC-S3-02 | Credential | JWT (`eyJ`) | Cleared |
| TC-S3-03 | Injection | `IGNORE PREVIOUS INSTRUCTIONS` | Cleared |
| TC-S3-04 | Clean | No patterns | `answer_text` unchanged |

---

## 7. Coverage Summary

| Criterion | Tests |
|-----------|-------|
| STTNode text passthrough | TC-STT-01, PB-7 |
| Error propagation (nodes return {}) | TC-STT-03, PB-8 |
| S-1 trust level declarations | TC-STT-05, TC-MAIN-02, PB-4 |
| S-2 PII gate | TC-S2-01–06 |
| S-3 output gate | TC-S3-01–04 |
| Emit trace events | PB-1 |
| Import isolation | PB-3 |
| State safety | PB-2, PB-5 |
| Full pipeline | TC-INT-01–03 |

---
*Test Specification: CMN-C1-113 · updated post-migration · 2026-06-23*
