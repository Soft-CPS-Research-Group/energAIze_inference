"""SQLite journal: pointer and successful activation receipt commit together."""
import hashlib
import json
import uuid
import time
from app.preparation.storage import PreparationError, canonical


class LifecycleStore:
    def __init__(self, store):
        self.store = store
        self.now = time.time
        with store.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS lifecycle_heads (singleton INTEGER PRIMARY KEY CHECK(singleton=1), context_id TEXT, activation_id TEXT);
                INSERT OR IGNORE INTO lifecycle_heads VALUES (1,NULL,NULL);
                CREATE TABLE IF NOT EXISTS installation_contexts (id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS context_updates (key_hash TEXT PRIMARY KEY, request TEXT NOT NULL, result TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS activation_operations (id TEXT PRIMARY KEY, key_hash TEXT UNIQUE NOT NULL, request TEXT NOT NULL, status TEXT NOT NULL, error TEXT);
                CREATE TABLE IF NOT EXISTS activations (id TEXT PRIMARY KEY, document TEXT NOT NULL);
            ''')
            columns = {row[1] for row in db.execute("PRAGMA table_info(activation_operations)")}
            if "expires_at" not in columns:
                db.execute("ALTER TABLE activation_operations ADD COLUMN expires_at REAL")
            error = canonical(dict(code="activation_interrupted", path="/", detail="Activation did not commit; retry with a new key after inspecting current state"))
            db.execute("UPDATE activation_operations SET status='uncertain',error=? WHERE status IN ('pending','running')", (error,))

    @staticmethod
    def heads_in(db):
        row = db.execute("SELECT context_id,activation_id FROM lifecycle_heads WHERE singleton=1").fetchone()
        return dict(row)

    def heads(self):
        with self.store.connection() as db:
            return self.heads_in(db)

    @staticmethod
    def check(heads, request):
        for field in ("context_id", "activation_id"):
            if heads[field] != request["expected_" + field]:
                raise PreparationError("precondition_failed", f"Current {field} differs from the expected value", "/expected_" + field, 409)

    def context(self, context_id):
        with self.store.connection() as db:
            row = db.execute("SELECT body FROM installation_contexts WHERE id=?", (context_id,)).fetchone()
        if row is None:
            raise PreparationError("context_not_found", "Installation context not found", "/context_id", 404)
        return json.loads(row[0])

    def declare(self, key, request):
        digest, body = hashlib.sha256(key.encode()).hexdigest(), canonical(request)
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT request,result FROM context_updates WHERE key_hash=?", (digest,)).fetchone()
            if prior:
                if prior[0] != body:
                    raise PreparationError("idempotency_conflict", "Key belongs to another context update", "/", 409)
                return json.loads(prior[1])
            heads = self.heads_in(db)
            self.check(heads, request)
            context_id = "ctx_" + uuid.uuid4().hex
            # Identical declarations keep the same fence and do not suspend serving.
            previous = db.execute("SELECT body FROM installation_contexts WHERE id=?", (heads["context_id"],)).fetchone()
            installation = canonical(request["installation"])
            if previous and previous[0] == installation:
                context_id = heads["context_id"]
            else:
                db.execute("INSERT INTO installation_contexts VALUES (?,?)", (context_id, installation))
                db.execute("UPDATE lifecycle_heads SET context_id=? WHERE singleton=1", (context_id,))
            result = dict(context_id=context_id, installation=request["installation"])
            db.execute("INSERT INTO context_updates VALUES (?,?,?)", (digest, body, canonical(result)))
            return result

    def submit(self, key, request):
        digest, body = hashlib.sha256(key.encode()).hexdigest(), canonical(request)
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT id,request FROM activation_operations WHERE key_hash=?", (digest,)).fetchone()
            if prior:
                if prior[1] != body:
                    raise PreparationError("idempotency_conflict", "Key belongs to another activation", "/", 409)
                return prior[0]
            self.check(self.heads_in(db), request)
            if db.execute("SELECT 1 FROM activation_operations WHERE status IN ('pending','running')").fetchone():
                raise PreparationError("activation_busy", "One activation is already admitted", "/", 429)
            operation_id = "act_" + uuid.uuid4().hex
            db.execute("INSERT INTO activation_operations(id,key_hash,request,status,expires_at) VALUES (?,?,?,'pending',?)",
                       (operation_id, digest, body, self.now() + request["timeout_seconds"]))
            return operation_id

    def operation(self, operation_id):
        with self.store.connection() as db:
            row = db.execute("SELECT * FROM activation_operations WHERE id=?", (operation_id,)).fetchone()
        if row is None:
            raise PreparationError("operation_not_found", "Activation operation not found", "/operation_id", 404)
        return dict(row)

    def claim(self, operation_id):
        with self.store.connection() as db:
            return db.execute("UPDATE activation_operations SET status='running' WHERE id=? AND status='pending'", (operation_id,)).rowcount == 1

    def fail(self, operation_id, error):
        # Never overwrite a successful commit, including after an ambiguous exception.
        with self.store.connection() as db:
            db.execute("UPDATE activation_operations SET status='failed',error=? WHERE id=? AND status='running'", (canonical(error), operation_id))

    def commit(self, operation_id, document):
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT request,status,expires_at FROM activation_operations WHERE id=?", (operation_id,)).fetchone()
            if row is None or row["status"] != "running":
                raise PreparationError("operation_not_running", "Activation is not running", "/operation_id", 409)
            if row["expires_at"] is not None and self.now() >= row["expires_at"]:
                raise PreparationError("activation_expired", "Activation deadline passed before publication", "/timeout_seconds", 409)
            request = json.loads(row["request"])
            self.check(self.heads_in(db), request)
            db.execute("INSERT INTO activations VALUES (?,?)", (operation_id, canonical(document)))
            db.execute("UPDATE lifecycle_heads SET activation_id=? WHERE singleton=1", (operation_id,))
            db.execute("UPDATE activation_operations SET status='succeeded',error=NULL WHERE id=?", (operation_id,))

    def activation(self, activation_id):
        with self.store.connection() as db:
            row = db.execute("SELECT document FROM activations WHERE id=?", (activation_id,)).fetchone()
        if row is None:
            raise PreparationError("activation_not_found", "Activation not found", "/activation_id", 404)
        return json.loads(row[0])
