# kkk-hermes-memory-zvec

A community **memory backend plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent)**, backed by [Zvec](https://github.com/zvec-ai) vector storage and local [Ollama](https://ollama.com) embeddings.

`memory-zvec` keeps your agent's long-term memory in an in-process vector
database (HNSW + RocksDB FTS) with **native hybrid retrieval** — one Zvec
`MultiQuery` call combining vector similarity, scalar filtering and full-text
search via `RRFReRanker`. Embeddings run locally with Ollama `bge-m3`
(1024-dim), so nothing leaves your machine.

This port's main focus is **lock governance for long-lived sessions**: stock
setups can deadlock or stall when a messaging-platform session and a
dashboard-embedded chat share the same collection. v1.1–v1.3.1 fix that class
of problems end to end (see [CHANGELOG.md](CHANGELOG.md)).

## Features

- **Native hybrid search** — vector + scalar filter + FTS in a single Zvec
  query with RRF re-ranking; scalar fields (`session_id`, `created_at`) use
  index-level pre-filtering, not post-ANN filtering.
- **Session-safe locking** — initialize-time write drain + backoff retry,
  idle lock-release watchdog (default 30s, reacts within ~5s), and a
  collection handle that survives session boundaries.
- **Drop-in compatible with `memory-lancedb`** — same 5 tool schemas
  (`vec_memory_add/search/list/delete/stats`) and the same
  prefetch / `sync_turn` / `on_session_end` hooks. Switching backends is a
  config edit plus a one-time data migration; existing 1024-dim vectors are
  reused without re-embedding.
- **Fast** — batch inserts of ~13k docs/s and sub-ms search under 100k docs
  in the author's testing (no formal benchmark published).
- **Local-first** — Ollama embeddings, in-process storage, no external
  services beyond the Ollama endpoint you configure.

### Search modes

`vec_memory_search` exposes Zvec's dual retrieval paths — three selectable
modes, so you can search by meaning, by exact text, or both at once:

| `mode` | What it does | Best for |
|---|---|---|
| `hybrid` (default) | Vector similarity + full-text keyword in **one** query, fused by RRF re-ranking | Natural-language questions |
| `vector` | Semantic similarity only | Paraphrased / fuzzy recall ("what did we decide about that pricing issue?") |
| `keyword` | Full-text only (RocksDB FTS, jieba tokenizer — Chinese-aware) | Exact terms, names, error codes |

Scalar filters ride along in **every** mode: `session_id` and
`after_timestamp` / `before_timestamp` narrow the candidate set via
index-level pre-filtering (applied before ANN traversal, not after).

## Requirements

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) with a working
  plugin directory (`~/.hermes/plugins/`)
- An **embedding model** served over an Ollama-compatible API (`/api/embed`):
  local [Ollama](https://ollama.com) with `bge-m3` (1024-dim, the default), or
  any hosted/online endpoint implementing the same API — just point `base_url`
  at it. The endpoint's vector dimension must match `vector_dim`.
- Python 3.11+ (the Hermes venv); dependencies (`zvec>=0.5.0,<0.7`, `numpy`,
  `requests`) are installed automatically from `plugin.yaml`

## Install

1. Copy the plugin into your Hermes plugin directory **as `memory-zvec`**:

   ```bash
   git clone https://github.com/kuntao2011/kkk-hermes-memory-zvec.git
   cp -r kkk-hermes-memory-zvec ~/.hermes/plugins/memory-zvec
   ```

   The directory name must be `memory-zvec` — memory providers are resolved
   by directory name (bundled first, then `$HERMES_HOME/plugins/`).

2. Point your profile at it in `$HERMES_HOME/config.yaml`:

   ```yaml
   memory:
     provider: memory-zvec

   plugins:
     memory-zvec:
       base_url: http://localhost:11434
       embedding_model: bge-m3:latest
       vector_dim: 1024
       zvec_dir: $HERMES_HOME/memory/zvec_memory   # override recommended — the built-in default points at the author's own dir layout
       collection_name: memories
   ```

3. Restart the gateway. Dependencies are installed automatically on the next
   start / `hermes update`.

> **Multi-profile:** Hermes profiles are isolated by `$HERMES_HOME` — each
> profile keeps its own plugin copy, config and data. To use the backend in
> another profile, repeat the install + config there; nothing is shared or
> synced between profiles.

## Configuration

| Key | Default | Description |
|---|---|---|
| `base_url` | `http://localhost:11434` | Ollama server URL |
| `embedding_model` | `bge-m3:latest` | Embedding model available in Ollama |
| `vector_dim` | `1024` | Embedding vector dimension (bge-m3 = 1024) |
| `zvec_dir` | `$HERMES_HOME/记忆数据库/zvec_memory` | Zvec collection directory (profile-scoped; any path works) |
| `collection_name` | `memories` | Zvec collection name |
| `batch_size` | `32` | Max texts per embedding batch |
| `search_top_k` | `5` | Default top-k results per search |
| `min_content_len` | `50` | Skip content shorter than this (chars) |
| `vector_weight` | `0.7` | Hybrid search weight for the vector branch |
| `fts_weight` | `0.3` | Hybrid search weight for the FTS branch |
| `enable_hnsw_optimize` | `true` | Call `collection.optimize()` after large bulk inserts |

Environment knobs: `HERMES_MEMORY_ZVEC_IDLE_RELEASE_S` — seconds of idleness
before the watchdog releases a collection lock (default 30);
`HERMES_MEMORY_ZVEC_INIT_DRAIN_S` — max wait draining the previous session's
in-flight writes at initialize (default 15); `HERMES_MEMORY_ZVEC_EXIT_DRAIN_S`
— the same at process exit.

## Migrating from memory-lancedb

Existing 1024-dim vectors are reused as-is — no re-embedding needed. A full
migration, health-check and repair playbook lives in
[kkk-hermes-zvec-memory-migration](https://github.com/kuntao2011/kkk-hermes-zvec-memory-migration).

## Relation to upstream

This is an independent community plugin, **not** part of the Hermes Agent
tree. Per upstream policy, new memory providers ship as standalone repos that
implement the same `MemoryProvider` ABC and are discovered through the same
path. Thanks to [Nous Research](https://github.com/NousResearch) for the
plugin architecture and to the Zvec team for the storage engine.

## License

[MIT](LICENSE) — the upstream Hermes Agent plugin interface is MIT as well.


## Experimental fork enhancements: selective facts and automatic recall

This fork preserves the original full-turn behavior by default. To **opt in**
to selective long-term facts, add the following keys under the existing
`plugins.memory-zvec` section in `$HERMES_HOME/config.yaml`:

```yaml
plugins:
  memory-zvec:
    memory_write_mode: facts
    extraction_model: qwen3.5:9b
    extraction_timeout_s: 45
    prefetch_mode: hybrid
    prefetch_top_k: 5
```

Keep the existing `base_url`, `embedding_model`, `vector_dim`, `zvec_dir`
and `collection_name` values. Do NOT replace the entire config section.

**Selective mode:** Each automatic sync/recovery hook submits the user's
message (not the assistant response) to local Ollama `/api/generate` with
JSON-only extraction instructions. It stores **one concise, user-sourced
fact** as role `fact` per source turn, or stores nothing when there is no
durable fact. Fictional examples, unsupported assistant guesses, and
questions should be skipped. A deterministic 64-character SHA256 document
ID prevents duplicate writes across `sync_turn` and `on_session_end`.
No remote extraction API is used. You must already have the configured
generation model available in Ollama (`ollama list`).

Extraction is imperfect because it is model-driven. Inspect saved facts
for accuracy and **avoid storing secrets or sensitive data**. The system
prompt instructs exclusion but is not a security boundary. On an Ollama
error, selective mode logs a warning and retries at session end; it **does
not** save the raw conversation as a hidden fallback. For guaranteed
transcript retention, use `memory_write_mode: turns`.

**Recall:** Automatic prefetch defaults to cross-session `hybrid` mode
(semantic vector + full-text), up to five results, with vector-only
fallback if hybrid yields no results. Override `prefetch_mode: vector`
to restore previous ranking. Existing Zvec records are left unchanged;
switching modes does not migrate or delete existing memory.

The native test suite runs with real Zvec 0.6.0 in isolated temporary
collections. No test connects to your production database or local Ollama.


**Session end:** In selective mode only the two most recent turns are retried
by default (`facts_session_end_max_turns: 2`, 0 to 10). Normal `sync_turn`
processes each turn as it happens. This avoids repeatedly running a large
local generation model over an entire chat when you close the session.
Long conversations where `sync_turn` was unavailable may lose older facts
in selective mode. To retain every turn reliably, keep `turns` mode.

The provider's memory status and tool descriptions use the configured
embedding model and dimensionality; you can keep `nomic-embed-text` with
`vector_dim: 768` while using `qwen3.5:9b` for fact extraction.
