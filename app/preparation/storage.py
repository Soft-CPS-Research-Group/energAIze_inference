"""Transactional, immutable revision storage and durable operation receipts."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class PreparationError(Exception):
    def __init__(self, code, detail, path="/", status=422):
        super().__init__(detail)
        self.code, self.detail, self.path, self.status = code, detail, path, status

    def document(self):
        return {"code": self.code, "path": self.path, "detail": self.detail}


class LocalStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "preparation.sqlite3"
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS configurations (revision TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS identity (name TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS operations (
                    id TEXT PRIMARY KEY, key_hash TEXT UNIQUE NOT NULL, request TEXT NOT NULL,
                    status TEXT NOT NULL, candidate TEXT, error TEXT);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def put(self, revision, document):
        body = canonical(document)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM configurations WHERE revision=?", (revision,)).fetchone()
            if row:
                if row[0] != body:
                    raise PreparationError("revision_conflict", "Revision already has different content", "/revision", 409)
                return False
            db.execute("INSERT INTO configurations VALUES (?,?)", (revision, body))
            return True

    def bind_site(self, site_id):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT value FROM identity WHERE name='site'").fetchone()
            if old and old[0] != site_id:
                raise ValueError("Preparation store belongs to a different installation")
            db.execute("INSERT OR IGNORE INTO identity VALUES ('site',?)", (site_id,))

    def configuration(self, revision):
        with self.connection() as db:
            row = db.execute("SELECT body FROM configurations WHERE revision=?", (revision,)).fetchone()
        if row is None:
            raise PreparationError("configuration_not_found", "Configuration revision not found", "/configuration_revision", 404)
        return json.loads(row[0])

    def submit(self, key, request, max_pending=4, max_stored=64):
        digest = hashlib.sha256(key.encode()).hexdigest()
        body = canonical(request)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT * FROM operations WHERE key_hash=?", (digest,)).fetchone()
            if existing:
                if existing["request"] != body:
                    raise PreparationError("idempotency_conflict", "Idempotency key was used for another request", "/", 409)
                return existing["id"]
            if db.execute("SELECT count(*) FROM operations WHERE status IN ('pending','running')").fetchone()[0] >= max_pending:
                raise PreparationError("preparation_queue_full", "Preparation admission limit reached", "/", 429)
            if db.execute("SELECT count(*) FROM operations").fetchone()[0] >= max_stored:
                raise PreparationError("candidate_retention_limit", "Candidate history limit reached; provision a larger limit or a new store", "/", 507)
            operation_id = uuid.uuid4().hex
            db.execute("INSERT INTO operations(id,key_hash,request,status) VALUES (?,?,?,'pending')", (operation_id, digest, body))
            return operation_id

    def operation(self, operation_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM operations WHERE id=?", (operation_id,)).fetchone()
        if row is None:
            raise PreparationError("operation_not_found", "Operation not found", "/operation_id", 404)
        return dict(row)

    def claim(self, operation_id):
        with self.connection() as db:
            return db.execute("UPDATE operations SET status='running' WHERE id=? AND status='pending'", (operation_id,)).rowcount == 1

    def finish(self, operation_id, candidate=None, error=None):
        with self.connection() as db:
            db.execute("UPDATE operations SET status=?,candidate=?,error=? WHERE id=?", (
                "failed" if error else "succeeded",
                canonical(candidate) if candidate else None,
                canonical(error) if error else None, operation_id,
            ))

    def recover(self):
        # Called only after obtaining the process-wide exclusive owner lock.
        error = canonical({"code": "preparation_interrupted", "path": "/", "detail": "Process stopped before completion; submit a new idempotency key to revalidate"})
        with self.connection() as db:
            db.execute("UPDATE operations SET status='uncertain',error=? WHERE status IN ('pending','running')", (error,))

    def counts(self):
        with self.connection() as db:
            latest = db.execute("SELECT id, status FROM operations ORDER BY rowid DESC LIMIT 1").fetchone()
            return {
                "configurations": db.execute("SELECT count(*) FROM configurations").fetchone()[0],
                "completed_candidates": db.execute("SELECT count(*) FROM operations WHERE candidate IS NOT NULL").fetchone()[0],
                "latest_operation": {"operation_id": latest["id"], "status": latest["status"]} if latest else None,
            }
