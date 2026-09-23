# CMN-C1-113 — Design Document

## Metadata

| Field | Value |
|-------|-------|
| Template ID | CMN-C1-113 |
| Agent Name | VoiceActivatedFAQAgent |
| L1 Base | `AgentBaseGraph` |
| SDK | `agenticstar-agentcore==1.0.0rc1` |
| Category | Cat 1 (CMN — cross-industry: RET, FIN, TEL, HCR, SVC) |
| Stage | stg-complete |

---

## Overview

Voice-Activated Customer FAQ Agent (Voice Loop).
Accepts audio input (file or text fallback), transcribes speech to text via Whisper,
retrieves relevant FAQ answers from a knowledge base, generates a natural-language reply
via LLM, and synthesises an audio response via VoxCPM2.

**Sprint 2 scope:** text/file input + TTS output. Live telephony and WebRTC deferred to Sprint 3.

**Statutory context:** バリアフリー法 2024 — expanded digital accessibility obligations for enterprises
serving Japan's 65+ demographic (29% of population, rising to 35%+ by 2040). Voice-first
interfaces are a primary compliance path.

---

## Architecture — 3-Slot SDK Pipeline

`VoiceFAQGraph` inherits `AgentBaseGraph` and maps 4 conceptual nodes into 3 SDK slots:

```
START → initialize → pre_process → main → post_process → finalize → END

  pre_process  → STTNode          (audio/text → transcript)
  main         → MainNode         (RetrieverNode → GeneratorNode, composite FunctionNode)
  post_process → TTSNode          (answer_text → audio_output_path)
```

Runtime config values from `config/config.yaml` are injected via `register_nodes()`:
```python
self._nodes["pre_process"]  = STTNode()
self._nodes["main"]         = MainNode(retriever_top_k=..., llm_model=...)
self._nodes["post_process"] = TTSNode(tts_chunk_max_chars=...)
```

Text input is passed as `user_input` at invocation time (from `POST /invoke`):
```python
agent.invoke(
    user_input=text,
    ctx=ctx,
)
```

Audio input never reaches the graph as bytes: `POST /invoke/audio` in `src/api/server.py`
reads the upload, validates it, transcribes it with Whisper, and calls `agent.invoke()`
with the transcript as `user_input` plus `input_context={"input_mode": "audio",
"audio_response_requested": True}` (ADR-005 — raw audio bytes must never enter graph
state, which has to stay msgpack-serializable).

---

## State Schema (`src/schemas/state.py`)

`VoiceFAQState` extends `AgentState` — flat TypedDict, msgpack-safe, no Pydantic/dataclass.

### Input fields
| Field | Type | Description |
|-------|------|-------------|
| `text_input` | `Optional[str]` | Text question, or the Whisper transcript for an audio request — transcription happens in `src/api/server.py` before `agent.invoke()`; no audio path/bytes ever enter state. |

### Processing fields
| Field | Type | Description |
|-------|------|-------------|
| `transcript` | `Optional[str]` | text_input passthrough. Set by STTNode (STTNode itself never calls Whisper — see Node Responsibilities below). |
| `stt_language` | `Optional[str]` | Always `"text_passthrough"` by the time state reaches STTNode's output. |
| `stt_confidence` | `Optional[float]` | Always 1.0 (text passthrough). |
| `retrieved_chunks` | `Optional[str]` | JSON list of {text, score, source}. Set by RetrieverNode. |
| `retrieval_score_max` | `Optional[float]` | Highest similarity score. Set by RetrieverNode. |
| `answer_text` | `Optional[str]` | LLM-generated answer. Set by GeneratorNode. |
| `answer_language` | `Optional[str]` | ISO 639-1 code for answer. Set by GeneratorNode. |
| `audio_output_path` | `Optional[str]` | Path to synthesised WAV. Set by TTSNode. |
| `audio_duration_sec` | `Optional[float]` | Audio duration in seconds. Set by TTSNode. |

### Output fields
| Field | Type | Description |
|-------|------|-------------|
| `formatted_output` | `Optional[dict[str, Any]]` | SDK get_output() result["output"]. |

### Error fields
| Field | Type | Description |
|-------|------|-------------|
| `error_message` | `Optional[str]` | First error encountered. Downstream nodes return {} when set. |

---

## Node Responsibilities

| Node | SDK Slot | Input | Output | Key logic |
|------|----------|-------|--------|-----------|
| `STTNode` | pre_process | text_input | transcript, stt_language, stt_confidence | Text passthrough only — Whisper transcription happens in `src/api/server.py`, ahead of `agent.invoke()`. S-1 trust gate. |
| `MainNode` | main | transcript | retrieved_chunks, retrieval_score_max, answer_text, answer_language | Composite: RetrieverNode → GeneratorNode |
| `TTSNode` | post_process | answer_text | audio_output_path, audio_duration_sec, formatted_output | VoxCPM2 synthesis |

### Error propagation
When `error_message` is set, nodes return `{}` immediately — LangGraph merges `{}` leaving
existing state unchanged. First error is preserved end-to-end.

### STTNode — text passthrough rule
STTNode copies `text_input` (or `user_input`) into `transcript` and sets
`stt_language = "text_passthrough"`, `stt_confidence = 1.0`. It never calls Whisper
itself — audio is transcribed earlier, in `src/api/server.py`'s `POST /invoke/audio`
handler, before `agent.invoke()` is ever called.

### RetrieverNode — no-result handling
When `retrieval_score_max < 0.45`, RetrieverNode still writes `retrieved_chunks` and
`retrieval_score_max`. GeneratorNode injects a disclaimer. No error set.

---

## Security Controls

| Layer | Implementation |
|-------|---------------|
| S-1 | `VoiceFAQGraph.required_trust_level = TrustLevel.INTERNAL`. SDK `BaseNode.__call__()` enforces before every execute(). |
| S-2 | `VoiceFAQGraph._extra_security_gate_input()`: PII scan on `text_input` and `transcript` (credit card, My Number, email, JP phone, JP postal code). Sets `error_message` + `status=ERROR` on detect. |
| S-3 | `VoiceFAQGraph._extra_security_gate_output()`: credential scan + injection scan on `answer_text`. Redacts field and sets `error_message` on detect. |
| S-4 | `emit_trace_event()` from `shared.utils.audit_logger` called in every node. `transcript`/`text_input` never logged verbatim — only `char_count`. |
| S-5 | `required_trust_level = TrustLevel.INTERNAL` on STTNode and MainNode. Inner nodes use `TrustLevel.ANONYMOUS`. |

---

## Configuration

### `config/agent.yaml` — SDK manifest
```yaml
template_id: CMN-C1-113
name: cmn-c1-113
version: "1.0.0"
category: 1
required_trust_level: INTERNAL
requires:
  extras: [openai]
  secrets:
    - OPENAI_API_KEY
    - QDRANT_URL
```

### `config/config.yaml` — runtime parameters
```yaml
max_retry: 1
memory_enabled: false
timeout_s: 120

stt_model: base
retriever_top_k: 5
retriever_score_threshold: 0.45
qdrant_collection: faq_kb
llm_model: gpt-4o-mini
tts_chunk_max_chars: 200
```

Config values are loaded by SDK into `self.config` and injected into nodes via `register_nodes()`.
Per-request overrides can be passed via `input_context`.

---

## Definition of Done

- [x] `pyproject.toml` — `agenticstar-platform` removed; domain deps pinned exactly
- [x] `config/agent.yaml` — SDK manifest format; `required_trust_level: INTERNAL`
- [x] `config/config.yaml` — runtime params wired into nodes via register_nodes()
- [x] `src/schemas/state.py` — `VoiceFAQState(AgentState)`, flat TypedDict, `formatted_output` added
- [x] 4 nodes migrated to `FunctionNode`; `execute(state) -> dict`; `emit_trace_event()`
- [x] `src/graph/graph.py` — `VoiceFAQGraph(AgentBaseGraph)` 3-slot SDK; S-2/S-3 via `_extra_*` hooks
- [x] `src/api/server.py` — `bound_secrets` + `provision_secrets`; `POST /invoke`, `GET /health`
- [x] Legacy pre-wheel provisioning and CoE-distributed reference files removed
- [x] CI config wired via the shared scaffold include
- [x] Tests migrated to SDK pattern (PB set1/set2 rewritten)
- [x] CI pipeline green

---
*Design Specification: CMN-C1-113 · updated post-migration · 2026-06-23*
