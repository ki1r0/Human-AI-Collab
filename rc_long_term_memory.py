from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rc_config import LTM_SQLITE_PATH, MEM0_API_KEY, MEMORY_ROOT


@dataclass
class MemoryItem:
    text: str
    metadata: Dict[str, Any]
    score: float | None = None


class LongTermMemory:
    """Long-term memory interface with Mem0 adapter + sqlite fallback.

    This object is thread-safe and can be called from the cognition thread.
    """

    def __init__(self, *, path: str = LTM_SQLITE_PATH, logger=print) -> None:
        self._log = logger
        self._lock = threading.Lock()
        self.backend = "sqlite"

        os.makedirs(MEMORY_ROOT, exist_ok=True)
        sqlite_fallback = _SqliteAdapter(path=path, logger=logger)
        mem0_key = self._resolve_mem0_key()
        if mem0_key:
            key_tail = mem0_key[-6:] if len(mem0_key) >= 6 else mem0_key
            self._log(f"[INFO] LongTermMemory: configured MEM0_API_KEY=***{key_tail}")
            # Force deterministic key source before importing Mem0 internals.
            os.environ["MEM0_API_KEY"] = mem0_key
            os.environ["OPENAI_API_KEY"] = ""

        self._impl: Any
        if not mem0_key:
            self._impl = sqlite_fallback
            self.backend = "sqlite"
            self._log(
                "[WARN] LongTermMemory: MEM0_API_KEY not set; using sqlite fallback "
                f"at {path}."
            )
            return

        try:
            # Use Mem0 cloud client only. Do not fall back to OpenAI-key based SDK paths.
            from mem0 import MemoryClient  # type: ignore

            self._impl = _Mem0Adapter(MemoryClient, fallback=sqlite_fallback, logger=logger, use_client=True)
            self.backend = "mem0_client"
            self._log("[INFO] LongTermMemory: using Mem0 backend (MemoryClient).")
            return
        except Exception as exc:
            self._impl = sqlite_fallback
            self.backend = "sqlite"
            self._log(
                f"[WARN] LongTermMemory: Mem0 client unavailable ({exc}); using sqlite fallback at {path}."
            )
            return

    @staticmethod
    def _resolve_mem0_key() -> str:
        """Resolve Mem0 API key with deterministic precedence.

        Order:
        1) rc_config.py file content (fresh on-disk value; avoids stale shell/env drift)
        2) MEM0_API_KEY env (runtime override fallback)
        3) imported rc_config.MEM0_API_KEY (fallback)
        """
        try:
            cfg_path = os.path.join(os.path.dirname(__file__), "rc_config.py")
            with open(cfg_path, "r", encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'^\s*MEM0_API_KEY\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
            if m:
                file_key = (m.group(1) or "").strip()
                if file_key:
                    return file_key
        except Exception:
            pass

        env_key = (os.getenv("MEM0_API_KEY", "") or "").strip()
        if env_key:
            return env_key

        return (MEM0_API_KEY or "").strip()

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        metadata = dict(metadata or {})
        if not text or not text.strip():
            return
        with self._lock:
            self._impl.add(text.strip(), metadata)

    def retrieve(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        query = (query or "").strip()
        if not query:
            return []
        with self._lock:
            return self._impl.retrieve(query, top_k=top_k)

    def end_episode(self, episode_log: Dict[str, Any]) -> None:
        """Summarize an episode and store it.

        In sqlite fallback, we store a compact textual record. If Mem0 is present, we store as well.
        """
        with self._lock:
            self._impl.end_episode(episode_log)


class _SqliteAdapter:
    def __init__(self, *, path: str, logger=print) -> None:
        self._log = logger
        self._path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self._path), exist_ok=True)

        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._has_fts = False
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts REAL NOT NULL,
              text TEXT NOT NULL,
              metadata_json TEXT NOT NULL
            );
            """
        )
        # Best-effort FTS. Some sqlite builds might not include FTS5.
        try:
            cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text, content='memories');")
            self._has_fts = True
        except Exception:
            self._has_fts = False
        self._conn.commit()

    def add(self, text: str, metadata: Dict[str, Any]) -> None:
        ts = time.time()
        meta_json = json.dumps(metadata, ensure_ascii=True)
        cur = self._conn.cursor()
        cur.execute("INSERT INTO memories(ts, text, metadata_json) VALUES (?, ?, ?);", (ts, text, meta_json))
        rowid = cur.lastrowid
        if self._has_fts and rowid is not None:
            try:
                cur.execute("INSERT INTO memories_fts(rowid, text) VALUES (?, ?);", (rowid, text))
            except Exception:
                # Not critical; keep going.
                pass
        self._conn.commit()

    def retrieve(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        cur = self._conn.cursor()
        items: List[MemoryItem] = []
        if self._has_fts:
            try:
                # bm25 returns lower-is-better by default; we invert to higher-is-better score.
                cur.execute(
                    """
                    SELECT m.text, m.metadata_json, bm25(memories_fts) AS rank
                    FROM memories_fts
                    JOIN memories m ON m.id = memories_fts.rowid
                    WHERE memories_fts MATCH ?
                    ORDER BY rank ASC
                    LIMIT ?;
                    """,
                    (query, int(top_k)),
                )
                for text, meta_json, rank in cur.fetchall():
                    try:
                        meta = json.loads(meta_json)
                    except Exception:
                        meta = {}
                    items.append(MemoryItem(text=text, metadata=meta, score=float(-rank)))
                return items
            except Exception as exc:
                self._log(f"[WARN] sqlite FTS retrieve failed ({exc}); falling back to LIKE.")

        # Fallback: naive LIKE ranking.
        like = f"%{query}%"
        cur.execute(
            "SELECT text, metadata_json, ts FROM memories WHERE text LIKE ? ORDER BY ts DESC LIMIT ?;",
            (like, int(top_k)),
        )
        for text, meta_json, _ts in cur.fetchall():
            try:
                meta = json.loads(meta_json)
            except Exception:
                meta = {}
            items.append(MemoryItem(text=text, metadata=meta, score=None))
        return items

    def end_episode(self, episode_log: Dict[str, Any]) -> None:
        # Store a compact episode summary. Keep it simple and robust.
        try:
            summary = episode_log.get("summary")
            if not summary:
                # Best-effort compact summary.
                parts = []
                if "task" in episode_log:
                    parts.append(f"task={episode_log['task']}")
                if "result" in episode_log:
                    parts.append(f"result={episode_log['result']}")
                if "last_belief" in episode_log:
                    parts.append(f"belief={episode_log['last_belief']}")
                if "dialogue" in episode_log:
                    parts.append(f"dialogue={episode_log['dialogue']}")
                summary = " | ".join(parts) if parts else json.dumps(episode_log, ensure_ascii=True)
            self.add(str(summary), {"type": "episode", "ts": time.time()})
        except Exception as exc:
            self._log(f"[WARN] Failed to store episode summary: {exc}")


class _Mem0Adapter:
    """Thin adapter around Mem0's Memory to match our interface."""

    def __init__(self, MemoryCls, *, fallback: _SqliteAdapter, logger=print, use_client: bool = False) -> None:
        self._log = logger
        self._fallback = fallback
        self._disabled_reason = ""
        self._use_client = bool(use_client)
        # Use the configured key from rc_config as the source of truth for this demo.
        # This avoids stale shell env values from older sessions.
        mem0_key = LongTermMemory._resolve_mem0_key()
        self._mem0_key_tail = mem0_key[-6:] if mem0_key else ""
        if mem0_key:
            os.environ["MEM0_API_KEY"] = mem0_key
            # Guard against key cross-talk: force this process to use Mem0 key path only.
            try:
                del os.environ["OPENAI_API_KEY"]
            except Exception:
                pass
            os.environ["OPENAI_API_KEY"] = ""
        # Mem0 requires at least one identity field on add/retrieve in newer SDK versions.
        self._user_id = os.getenv("MEM0_USER_ID", "isaac_user").strip()
        self._agent_id = os.getenv("MEM0_AGENT_ID", "headcam_agent").strip()
        self._run_id = os.getenv("MEM0_RUN_ID", f"run_{int(time.time())}").strip()
        if self._use_client:
            if not mem0_key:
                raise RuntimeError("MEM0_API_KEY is required for Mem0 client backend.")
            self._mem = MemoryCls(api_key=mem0_key)
        else:
            self._mem = MemoryCls()

    def _is_auth_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        # OpenAI-style auth errors can be returned by misconfigured Mem0 provider chains.
        openai_style_key_error = (
            "openai.com/account/api-keys" in msg
            or "platform.openai.com/account/api-keys" in msg
            or ("incorrect api key provided" in msg)
            or ("invalid_api_key" in msg)
            or ("invalid api key" in msg)
        )
        has_api_key_phrase = ("api key" in msg) and (
            "incorrect" in msg or "invalid" in msg or "must be set" in msg or "missing" in msg
        )
        # Catch provider-specific auth formatting (OpenAI-style and generic 401 payloads).
        has_401_phrase = ("401" in msg) and (
            "api key" in msg or "unauthorized" in msg or "invalid_request_error" in msg or "forbidden" in msg
        )
        return (
            openai_style_key_error
            or "unauthorized" in msg
            or "error code: 401" in msg
            or "status code: 401" in msg
            or "invalid_request_error" in msg
            or has_api_key_phrase
            or has_401_phrase
        )

    def _is_mem0_identity_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            ("at least one of" in msg and "must be provided" in msg and ("user_id" in msg or "agent_id" in msg or "run_id" in msg))
            or "missing required identity" in msg
        )

    def _disable_mem0(self, reason: str) -> None:
        if self._disabled_reason:
            return
        self._disabled_reason = reason
        suffix = f" key=***{self._mem0_key_tail}" if self._mem0_key_tail else ""
        self._log(f"[WARN] Mem0 disabled ({reason}); falling back to sqlite memory.{suffix}")

    def _identity_kwargs(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        run_id = self._run_id
        if isinstance(metadata, dict):
            v = metadata.get("run_id")
            if isinstance(v, str) and v.strip():
                run_id = v.strip()
        return {"user_id": self._user_id, "agent_id": self._agent_id, "run_id": run_id}

    def add(self, text: str, metadata: Dict[str, Any]) -> None:
        if self._disabled_reason:
            self._fallback.add(text, metadata)
            return
        try:
            # Mem0's API differs across versions; handle common ones.
            if hasattr(self._mem, "add"):
                ident = self._identity_kwargs(metadata)
                add_attempts = (
                    lambda: self._mem.add(text, metadata=metadata, **ident),
                    lambda: self._mem.add(messages=text, metadata=metadata, **ident),
                    lambda: self._mem.add(text, metadata=metadata, user_id=ident["user_id"]),
                    lambda: self._mem.add(messages=text, metadata=metadata, user_id=ident["user_id"]),
                    lambda: self._mem.add(text, **ident),
                    lambda: self._mem.add(messages=text, **ident),
                    lambda: self._mem.add(text, user_id=ident["user_id"]),
                    lambda: self._mem.add(messages=text, user_id=ident["user_id"]),
                )
                last_exc = None
                done = False
                for fn in add_attempts:
                    try:
                        fn()
                        done = True
                        break
                    except Exception as exc:  # pragma: no cover - signature compatibility path
                        last_exc = exc
                if not done and last_exc is not None:
                    raise last_exc
            else:
                raise AttributeError("Mem0 Memory has no .add()")
        except Exception as exc:
            if self._is_auth_error(exc):
                self._disable_mem0(str(exc))
                self._fallback.add(text, metadata)
                return
            if self._is_mem0_identity_error(exc):
                self._disable_mem0(str(exc))
                self._fallback.add(text, metadata)
                return
            self._log(f"[WARN] Mem0 add failed: {exc}")
            self._fallback.add(text, metadata)

    def retrieve(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        if self._disabled_reason:
            return self._fallback.retrieve(query, top_k=top_k)
        try:
            ident = self._identity_kwargs()
            if hasattr(self._mem, "search"):
                search_attempts = (
                    lambda: self._mem.search(query=query, limit=int(top_k), **ident),
                    lambda: self._mem.search(query=query, top_k=int(top_k), **ident),
                    lambda: self._mem.search(query, limit=int(top_k), **ident),
                    lambda: self._mem.search(query, top_k=int(top_k), **ident),
                    lambda: self._mem.search(query=query, limit=int(top_k), user_id=ident["user_id"]),
                    lambda: self._mem.search(query=query, top_k=int(top_k), user_id=ident["user_id"]),
                    lambda: self._mem.search(query, limit=int(top_k), user_id=ident["user_id"]),
                    lambda: self._mem.search(query, top_k=int(top_k), user_id=ident["user_id"]),
                )
                res = None
                last_exc = None
                for fn in search_attempts:
                    try:
                        res = fn()
                        break
                    except Exception as exc:  # pragma: no cover - signature compatibility path
                        last_exc = exc
                if res is None and last_exc is not None:
                    raise last_exc
            elif hasattr(self._mem, "retrieve"):
                retrieve_attempts = (
                    lambda: self._mem.retrieve(query=query, top_k=int(top_k), **ident),
                    lambda: self._mem.retrieve(query=query, limit=int(top_k), **ident),
                    lambda: self._mem.retrieve(query, top_k=int(top_k), **ident),
                    lambda: self._mem.retrieve(query, limit=int(top_k), **ident),
                    lambda: self._mem.retrieve(query=query, top_k=int(top_k), user_id=ident["user_id"]),
                    lambda: self._mem.retrieve(query=query, limit=int(top_k), user_id=ident["user_id"]),
                    lambda: self._mem.retrieve(query, top_k=int(top_k), user_id=ident["user_id"]),
                    lambda: self._mem.retrieve(query, limit=int(top_k), user_id=ident["user_id"]),
                )
                res = None
                last_exc = None
                for fn in retrieve_attempts:
                    try:
                        res = fn()
                        break
                    except Exception as exc:  # pragma: no cover - signature compatibility path
                        last_exc = exc
                if res is None and last_exc is not None:
                    raise last_exc
            else:
                raise AttributeError("Mem0 Memory has no search/retrieve method")

            items: List[MemoryItem] = []
            for r in res or []:
                text = str(r.get("memory") or r.get("text") or r)
                meta = r.get("metadata") if isinstance(r, dict) else {}
                score = r.get("score") if isinstance(r, dict) else None
                items.append(MemoryItem(text=text, metadata=meta or {}, score=score))
            return items
        except Exception as exc:
            if self._is_auth_error(exc):
                self._disable_mem0(str(exc))
                return self._fallback.retrieve(query, top_k=top_k)
            if self._is_mem0_identity_error(exc):
                self._disable_mem0(str(exc))
                return self._fallback.retrieve(query, top_k=top_k)
            self._log(f"[WARN] Mem0 retrieve failed: {exc}")
            return self._fallback.retrieve(query, top_k=top_k)

    def end_episode(self, episode_log: Dict[str, Any]) -> None:
        if self._disabled_reason:
            self._fallback.end_episode(episode_log)
            return
        try:
            text = episode_log.get("summary") or json.dumps(episode_log, ensure_ascii=True)
            self.add(text, {"type": "episode"})
        except Exception as exc:
            self._log(f"[WARN] Mem0 end_episode failed: {exc}")
            self._fallback.end_episode(episode_log)
