"""Prepare off the publication lock, persist atomically, then publish one generation."""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import threading

from app.contracts.v2.safety import ConfigurationV2
from app.core.errors import ExecutionError
from app.preparation.storage import PreparationError
from .storage import LifecycleStore


@dataclass
class Generation:
    document: dict
    execution: object
    directory: Path
    readers: int = 0
    execution_lock: object = field(default_factory=threading.Lock)


def sync_tree(root, stop):
    """Durable bytes and directory entries before the journal may reference them."""
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directories = [p for p in root.rglob("*") if p.is_dir()] + [root]
    parent = root.parent
    while True:
        directories.append(parent)
        if parent == stop:
            break
        parent = parent.parent
    for path in directories:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class ActivationService:
    def __init__(self, preparation):
        self.preparation = preparation
        self.store = LifecycleStore(preparation.store)
        self.lock = threading.Condition(threading.RLock())
        self.capacity = threading.BoundedSemaphore(preparation.settings.max_concurrent_inferences)
        self.active = None
        self.retired = None
        self.recovery_error = None
        self.closed = False
        self.root = preparation.store.root / "generations"
        self.root.mkdir(exist_ok=True, mode=0o700)
        self.recover()

    def _configuration(self, candidate_id):
        candidate = self.preparation.candidate(candidate_id)
        if candidate["state"] != "ready":
            raise PreparationError("candidate_not_ready", "Activation requires a ready candidate", "/candidate_id", 409)
        config = self.preparation.configuration(candidate["configuration_revision"])
        if not isinstance(config, ConfigurationV2):
            raise PreparationError("activation_configuration_unsupported", "Active decisions require explicit v2 safety configuration", "/configuration_revision", 422)
        return config

    def _compatible(self, candidate_id, context_id):
        config = self._configuration(candidate_id)
        if config.installation.dict() != self.store.context(context_id):
            raise PreparationError("installation_changed", "Candidate installation differs from the declared current context; prepare against that context", "/candidate_id", 409)
        previous_id = self.store.heads()["activation_id"]
        if previous_id:
            previous = self.store.activation(previous_id)
            prior_config = self._configuration(previous["candidate_id"])
            if (prior_config.bundle.manifest_sha256 == config.bundle.manifest_sha256 and
                    (prior_config.bundle.inputs != config.bundle.inputs or prior_config.bundle.outputs != config.bundle.outputs)):
                raise PreparationError("interface_changed_without_bundle_evidence", "Changed algorithm semantics/order with the same manifest hash needs reviewed bundle evidence", "/bundle", 409)
        return config

    def declare(self, key, request):
        if request.installation.site_id != self.preparation.settings.installation_site_id:
            raise PreparationError("installation_mismatch", "Context belongs to another installation", "/installation/site_id")
        with self.lock:
            return self.store.declare(key, request.dict())

    def submit(self, key, request):
        with self.lock:
            if self.closed or self.recovery_error:
                raise PreparationError("activation_unavailable", "Resolve recovery failure before activating", "/", 503)
            # Replay is handled by storage before checking current fences. Candidate
            # structural readiness remains true even when its context is superseded.
            self._configuration(request.candidate_id)
            return self.store.submit(key, request.dict())

    def receipt(self, operation_id):
        row = self.store.operation(operation_id)
        return dict(operation_id=operation_id, status=row["status"],
                    resource_id=operation_id if row["status"] == "succeeded" else None,
                    error=json.loads(row["error"]) if row["error"] else None)

    def _publish(self, generation):
        previous = self.active
        self.active = generation
        if previous:
            if previous.readers:
                self.retired = previous
            else:
                shutil.rmtree(previous.directory, ignore_errors=True)

    def activate(self, operation_id):
        if not self.store.claim(operation_id):
            return
        directory = self.root / operation_id
        committed = False
        try:
            with self.preparation.work(wait=True):
                operation = self.store.operation(operation_id)
                if operation["expires_at"] is not None and self.store.now() >= operation["expires_at"]:
                    raise PreparationError("activation_expired", "Activation deadline passed before construction", "/timeout_seconds", 409)
                request = json.loads(operation["request"])
                with self.lock:
                    self.store.check(self.store.heads(), request)
                    if self.closed or self.retired is not None:
                        raise PreparationError("generation_busy", "Previous generation still has in-flight requests", "/", 429)
                    self._compatible(request["candidate_id"], request["expected_context_id"])
                execution = self.preparation.build_execution(request["candidate_id"], directory / "files")
                document = dict(activation_id=operation_id, context_id=request["expected_context_id"],
                    candidate_id=request["candidate_id"], previous_activation_id=request["expected_activation_id"],
                    activated_at=datetime.now(timezone.utc).isoformat(), execution=asdict(execution.context.identity))
                generation = Generation(document, execution, directory)
                sync_tree(directory, self.preparation.store.root)
                with self.lock:
                    # A context declaration may have arrived while loading. SQLite
                    # verifies both fences again in the same transaction as the pointer.
                    self._compatible(request["candidate_id"], request["expected_context_id"])
                    try:
                        self.store.commit(operation_id, document)
                        committed = True
                        self._publish(generation)
                    except PreparationError:
                        raise  # A transactional precondition failure preserves the old active.
                    except BaseException:
                        # The database commit could have succeeded despite a lost
                        # acknowledgement. Never keep serving a possibly stale pointer.
                        self.recovery_error = dict(code="publication_uncertain", path="/activation", detail="Restart to reconcile the durable activation journal")
                        raise
        except Exception as exc:
            error = exc.document() if isinstance(exc, (PreparationError, ExecutionError)) else dict(code="activation_failed", path="/", detail="Activation failed; inspect the operation and active state")
            self.store.fail(operation_id, error)
        finally:
            # A committed generation is the recovery source even if publication failed.
            try:
                durable = self.store.heads()["activation_id"] == operation_id
            except Exception:
                durable = True  # Preserve evidence on an ambiguous storage failure.
            if not committed and not durable:
                shutil.rmtree(directory, ignore_errors=True)

    def recover(self):
        heads = self.store.heads()
        activation_id = heads["activation_id"]
        # Orphan and superseded runtime snapshots are not rollback authority.
        for directory in self.root.iterdir():
            if directory.name != activation_id and directory.is_dir() and not directory.is_symlink():
                shutil.rmtree(directory)
        if activation_id:
            try:
                document = self.store.activation(activation_id)
                # A new physical context deliberately suspends the previous model.
                if heads["context_id"] == document["context_id"]:
                    self._compatible(document["candidate_id"], document["context_id"])
                    # Revalidation copies all bytes before constructing native engines.
                    # Keep the copied runtime files alive for the generation lifetime.
                    directory = self.root / activation_id
                    runtime_dir = directory / "runtime"
                    shutil.rmtree(runtime_dir, ignore_errors=True)
                    with self.preparation.work():
                        execution = self.preparation.build_execution(document["candidate_id"], runtime_dir,
                            source_directory=f"generations/{activation_id}/files")
                    # Report the version of the currently installed execution code.
                    document["execution"] = asdict(execution.context.identity)
                    self.active = Generation(document, execution, directory)
            except Exception:
                self.recovery_error = dict(code="recovery_failed", path="/activation", detail="Committed activation could not be revalidated; restore its files/providers and restart")

    def status(self):
        with self.lock:
            heads = self.store.heads()
            usable = bool(self.active and not self.closed and not self.recovery_error and
                          self.active.document["activation_id"] == heads["activation_id"] and
                          self.active.document["context_id"] == heads["context_id"])
            return {**heads, "ready": usable, "state": "ready" if usable else
                    "recovery_failed" if self.recovery_error else "context_changed" if heads["activation_id"] else "unconfigured",
                    "error": self.recovery_error, "activation": self.active.document if usable else None,
                    "retired_requests": self.retired.readers if self.retired else 0}

    def infer(self, request):
        if not self.capacity.acquire(blocking=False):
            raise PreparationError("inference_capacity", "Inference admission limit reached", "/", 429)
        generation = None
        try:
            with self.lock:
                heads = self.store.heads()
                self.store.check(heads, request.dict())
                if self.closed or self.recovery_error or not self.active or self.active.document["context_id"] != heads["context_id"] or self.active.document["activation_id"] != heads["activation_id"]:
                    raise PreparationError("active_unavailable", "No usable activation for the current installation", "/", 503)
                generation = self.active
                generation.readers += 1
            # Native and rule engines have no universal concurrency guarantee.
            # Serialize per generation; admission bounds the waiting request count.
            with generation.execution_lock:
                result = generation.execution.evaluate(request.features)
            result.pop("evaluation_only")
            result.update(schema_version="decision.v2-draft", cycle_id=request.cycle_id,
                activation={k: v for k, v in generation.document.items() if k != "execution"},
                providers=list(generation.execution.runtime.providers))
            return result
        finally:
            with self.lock:
                if generation:
                    generation.readers -= 1
                    if generation is self.retired and not generation.readers:
                        shutil.rmtree(generation.directory, ignore_errors=True)
                        self.retired = None
                self.lock.notify_all()
            self.capacity.release()

    def close(self):
        with self.lock:
            self.closed = True
            self.lock.wait_for(lambda: not (self.active and self.active.readers) and not self.retired)
            self.active = None
