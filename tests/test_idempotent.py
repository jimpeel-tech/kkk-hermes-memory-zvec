"""Isolated tests; never access the user's Hermes/Zvec production paths."""
import importlib.util
import sys
import tempfile
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent.parent
agent = types.ModuleType("agent")
agent.__path__ = []
provider_module = types.ModuleType("agent.memory_provider")
provider_module.MemoryProvider = type("MemoryProvider", (), {})
tools = types.ModuleType("tools")
tools.__path__ = []
registry = types.ModuleType("tools.registry")
registry.tool_error = lambda message: message
sys.modules["agent"] = agent
sys.modules["agent.memory_provider"] = provider_module
sys.modules["tools"] = tools
sys.modules["tools.registry"] = registry
spec = importlib.util.spec_from_file_location("memory_zvec_under_test", root / "__init__.py")
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)


class Status:
    def __init__(self, code=None, message=""):
        self._code, self._message = code, message
    def ok(self): return self._code is None
    def code(self): return self._code
    def message(self): return self._message


class Doc:
    def __init__(self, id, vectors, fields):
        self.id, self.vectors, self.fields = id, vectors, fields


class FakeZvec:
    Doc = Doc
    StatusCode = types.SimpleNamespace(ALREADY_EXISTS="ALREADY_EXISTS")


class Collection:
    def __init__(self):
        self.docs, self.fail_next, self.flushes = {}, False, 0
    def fetch(self, ids, *, include_vector=True):
        ids = [ids] if isinstance(ids, str) else ids
        return {i: self.docs[i] for i in ids if i in self.docs}
    def insert(self, doc):
        if self.fail_next:
            self.fail_next = False
            return Status("UNAVAILABLE", "simulated failure")
        if doc.id in self.docs:
            return Status("ALREADY_EXISTS", "already exists")
        self.docs[doc.id] = doc
        return Status()
    def flush(self): self.flushes += 1
    def optimize(self): pass


class WriteTests(unittest.TestCase):
    def setUp(self):
        self.coll = Collection()
        p = object.__new__(plugin.ZvecMemoryProvider)
        self.p = p
        p._config = {"embedding_model": "nomic-embed-text", "min_content_len": 50,
                     "enable_hnsw_optimize": False}
        p._insert_lock = threading.Lock()
        p._embedding_lock = threading.Lock()
        p._writes_since_optimize = 0
        p._session_id = "session-a"
        p._coll = self.coll
        p._read_only = False
        p._ensure_open = lambda: None
        p._prefetch_cache = {}
        p._prefetch_ttl = 30
        self.num_embeds = 0
        self.orig_zvec, self.orig_embed = plugin._get_zvec, plugin._ollama_embed_single
        plugin._get_zvec = lambda: FakeZvec
        def embed(*args):
            self.num_embeds += 1
            return np.ones(3, dtype=np.float32)
        plugin._ollama_embed_single = embed
        self.user = "Project Indigo Heron database question"
        self.answer = "PostgreSQL and Redis with three Kubernetes nodes"

    def tearDown(self):
        plugin._drain_pending_writes(timeout=5)
        plugin._get_zvec, plugin._ollama_embed_single = self.orig_zvec, self.orig_embed

    def store(self, *, sid="session-a", role="turn", user=None, answer=None):
        return self.p._write_turn_once(
            user_content=self.user if user is None else user,
            assistant_content=self.answer if answer is None else answer,
            sid=sid, role=role, ts=1000, meta="{}", coll=self.coll)

    def test_idempotent_skips_duplicate_embedding(self):
        self.assertTrue(self.store())
        self.assertFalse(self.store(role="session_end"))
        self.assertEqual((len(self.coll.docs), self.num_embeds), (1, 1))

    def test_normalization_and_different_sessions(self):
        self.assertTrue(self.store(user="  "+self.user+"\r\n", answer=self.answer+"\n"))
        self.assertFalse(self.store())
        self.assertTrue(self.store(sid="session-b"))
        self.assertEqual(len(self.coll.docs), 2)

    def test_session_end_recovers_and_is_repeatable(self):
        messages = [{"role": "user", "content": self.user},
                    {"role": "assistant", "content": self.answer}]
        self.p.on_session_end(messages)
        plugin._drain_pending_writes(timeout=5)
        self.assertEqual(len(self.coll.docs), 1)
        self.assertEqual(next(iter(self.coll.docs.values())).fields["role"], "session_end")
        self.p.on_session_end(messages)
        plugin._drain_pending_writes(timeout=5)
        self.assertEqual(len(self.coll.docs), 1)

    def test_turn_and_session_end_share_document(self):
        self.p.sync_turn("  "+self.user, self.answer+"\n", session_id="session-a")
        plugin._drain_pending_writes(timeout=5)
        self.assertEqual(len(self.coll.docs), 1)
        self.p.on_session_end([{"role": "user", "content": self.user},
                               {"role": "assistant", "content": self.answer}])
        plugin._drain_pending_writes(timeout=5)
        self.assertEqual(len(self.coll.docs), 1)

    def test_concurrency(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.store(), range(20)))
        self.assertEqual((results.count(True), len(self.coll.docs), self.num_embeds), (1, 1, 1))

    def test_failed_insert_fails_loudly_then_recovers(self):
        self.coll.fail_next = True
        with self.assertRaisesRegex(RuntimeError, "simulated failure"):
            self.store()
        self.assertTrue(self.store())
        self.assertEqual(len(self.coll.docs), 1)

    def test_prefetch_ignores_session_filter(self):
        seen, event = [], threading.Event()
        def fake_search(query, top_k, session_id):
            seen.append(session_id)
            event.set()
            return []
        self.p._do_search = fake_search
        self.p.prefetch("indigo heron", session_id="different")
        self.p.queue_prefetch("other query", session_id="different")
        self.assertTrue(event.wait(timeout=3))
        self.assertEqual(seen[0], "")
        # Both prefetch implementations are intentionally global:
        self.assertEqual(plugin._turn_doc_id("s", self.user, self.answer),
                         plugin._turn_doc_id("s", self.user, self.answer))


class NativeApiTests(unittest.TestCase):
    def test_zvec_06_native_status_and_fetch(self):
        try:
            import zvec
        except ImportError:
            self.skipTest("requires zvec==0.6.0")
        with tempfile.TemporaryDirectory() as tmp:
            schema = plugin._build_memories_schema(vector_dim=3, collection_name="test_memories")
            coll = zvec.create_and_open(str(Path(tmp) / "test_memories"), schema)
            try:
                doc = zvec.Doc(
                    id="test-id",
                    vectors={"vector": [0.1, 0.2, 0.3]},
                    fields={"content": "a real Zvec test", "role": "turn",
                            "session_id": "s", "created_at": 1.0, "metadata": "{}"},
                )
                self.assertTrue(coll.insert(doc).ok())
                self.assertIn("test-id", coll.fetch(ids="test-id", include_vector=False))
                status = coll.insert(doc)
                self.assertFalse(status.ok())
                self.assertEqual(status.code(), zvec.StatusCode.ALREADY_EXISTS)
            finally:
                del coll

if __name__ == "__main__":
    unittest.main()
