"""Prepare and persist candidates independently of the active inference store."""

import fcntl
import json
import os
import threading
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from app.adapters.runtime_factory import ArtifactRuntimeFactory
from app.contracts.v1.execution import CompatibilityReport
from app.contracts.v2.safety import parse_configuration
from app.core.contracts import PipelineRecord
from app.logging import get_logger
from app.preparation.assessment import check_manifest, context_issues, impact, issue
from app.preparation.bundle import BundleSnapshot, read_json
from app.preparation.models import LocalBundle, PrepareCandidate
from app.preparation.storage import LocalStore, PreparationError
from app.services.pipeline import InferencePipeline


class PreparationService:
    def __init__(self, settings):
        self.settings = settings
        self.work_gate = threading.Lock()
        root = Path(settings.configuration_store_dir).resolve()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = (root / "owner.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.store = LocalStore(root)
            self.store.bind_site(settings.installation_site_id)
            self.store.recover()
            # Interrupted evaluation copies are temporary, never recovery authority.
            for path in root.glob("evaluation-*"):
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
            from app.lifecycle.service import ActivationService
            self.lifecycle = ActivationService(self)
        except Exception:
            self.lock.close()
            raise

    def close(self):
        with self.work_gate:
            self.lifecycle.close()
            self.lock.close()

    @contextmanager
    def work(self, wait=False):
        if not self.work_gate.acquire(blocking=wait):
            raise PreparationError("runtime_builder_busy", "Another bounded runtime construction is in progress", "/runtime", 429)
        try:
            # Conservative reservation for one maximum-size snapshot. Builders
            # are serialized, so another local operation cannot consume it.
            used = 0
            try:
                for folder in ("candidates", "generations"):
                    for directory, _, files in os.walk(self.store.root / folder):
                        for name in files:
                            try:
                                used += os.stat(Path(directory) / name, follow_symlinks=False).st_size
                            except FileNotFoundError:
                                pass  # An in-flight retired generation just drained.
            except OSError as exc:
                raise PreparationError("snapshot_storage_unavailable", "Cannot inspect private snapshot storage", "/runtime", 503) from exc
            if used + self.settings.max_candidate_bytes > self.settings.max_snapshot_storage_bytes:
                raise PreparationError("snapshot_storage_full", "Insufficient configured snapshot budget for another candidate", "/runtime", 507)
            yield
        finally:
            self.work_gate.release()

    def put(self, revision, config):
        if config.revision != revision:
            raise PreparationError("revision_mismatch", "Body and path revision differ", "/revision")
        if config.installation.site_id != self.settings.installation_site_id:
            raise PreparationError("installation_mismatch", "Configuration belongs to another installation", "/installation/site_id")
        return self.store.put(revision, config.dict())

    def configuration(self, revision):
        return parse_configuration(self.store.configuration(revision))

    def submit(self, key, request):
        self.configuration(request.configuration_revision)
        if request.baseline_configuration_revision is not None:
            self.configuration(request.baseline_configuration_revision)
        return self.store.submit(key, request.dict(), self.settings.max_pending_preparations, self.settings.max_stored_candidates)

    def receipt(self, operation_id):
        if operation_id.startswith("act_"):
            return self.lifecycle.receipt(operation_id)
        row = self.store.operation(operation_id)
        return dict(operation_id=operation_id, status=row["status"],
                    resource_id=operation_id if row["candidate"] else None,
                    error=json.loads(row["error"]) if row["error"] else None)

    def candidate(self, candidate_id):
        row = self.store.operation(candidate_id)
        if row["candidate"]:
            return json.loads(row["candidate"])
        return dict(candidate_id=candidate_id, state=row["status"], compatibility=None,
                    error=json.loads(row["error"]) if row["error"] else None)

    def prepare(self, operation_id):
        # Claim before waiting: duplicate HTTP retries never add more queued work.
        if not self.store.claim(operation_id):
            return
        try:
            with self.work(wait=True):
                self._prepare(operation_id)
        except PreparationError as exc:
            self.store.finish(operation_id, error=exc.document())

    def _prepare(self, operation_id):
        try:
            request = PrepareCandidate.parse_raw(self.store.operation(operation_id)["request"])
            config = self.configuration(request.configuration_revision)
            baseline = self.configuration(request.baseline_configuration_revision) if request.baseline_configuration_revision else None
            problems = context_issues(config)
            if baseline is not None and baseline.bundle.manifest_sha256 == config.bundle.manifest_sha256 and (baseline.bundle.inputs != config.bundle.inputs or baseline.bundle.outputs != config.bundle.outputs):
                problems.append(issue("interface_changed_without_bundle_evidence", "/bundle", "Changed algorithm semantics/order with the same manifest hash requires reviewed bundle evidence", "bundle"))
            snapshot = None
            validated_providers = []
            checks = ["site_and_bindings", "signal_meanings_and_access", "controlled_channel_coverage", "adapter_and_safety_capabilities"]
            try:
                snapshot = BundleSnapshot(self.settings.allowed_bundle_root, request.bundle,
                                          self.store.root / "candidates" / operation_id / "files",
                                          self.settings.max_candidate_bytes)
                manifest = snapshot.collect(config.bundle.manifest_sha256)
                checks.append("bundle_paths_and_hashes")
                artifact, width, names = check_manifest(config, manifest, snapshot)
                checks.append("manifest_interface_and_encoders")
                if not problems:
                    try:
                        record = self._construct(config, manifest, artifact, width, names, snapshot)
                        validated_providers = list(record.pipeline.agent.providers)
                        del record
                    except PreparationError:
                        raise
                    except Exception as exc:
                        raise PreparationError("runtime_construction_failed", "The isolated runtime could not be constructed", "/runtime") from exc
                    checks.append("isolated_runtime_construction")
            except PreparationError as exc:
                problems.append(issue(exc.code, exc.path, exc.detail, "bundle"))
            except (ValidationError, ValueError, TypeError, KeyError) as exc:
                problems.append(issue("bundle_contract_invalid", "/bundle", str(exc), "bundle"))
            status = "compatible"
            if problems:
                status = "insufficient_information" if all(p["code"].endswith("_unknown") for p in problems) else "incompatible"
            report = CompatibilityReport(schema_version="compatibility.v1-draft", configuration_revision=config.revision,
                                         status=status, checks_performed=checks, issues=problems)
            candidate = dict(candidate_id=operation_id, configuration_revision=config.revision,
                             state="ready" if status == "compatible" else "rejected", compatibility=report.dict(),
                             impact=impact(baseline, config), file_hashes=snapshot.hashes if snapshot else {},
                             requested_onnx_providers=list(self.settings.onnx_execution_providers),
                             validated_providers=validated_providers,
                             requires_revalidation_before_activation=True)
            self.store.finish(operation_id, candidate=candidate)
        except Exception:
            get_logger().exception("candidate.preparation_failed", operation_id=operation_id)
            self.store.finish(operation_id, error=dict(code="preparation_failed", path="/", detail="Candidate preparation failed; active state was not modified"))

    def _construct(self, config, manifest, artifact, width, names, snapshot):
        if (artifact.format or "onnx") == "onnx":
            import onnxruntime as ort
            if not self.settings.onnx_execution_providers or any(p not in ort.get_available_providers() for p in self.settings.onnx_execution_providers):
                raise PreparationError("provider_unavailable", "A requested ONNX provider is unavailable", "/runtime/providers")
        alias_path = snapshot.destination / snapshot.alias if snapshot.alias else None
        from app.adapters.bundles import _load_alias_overrides
        pipeline = InferencePipeline(manifest, snapshot.destination, artifact.agent_index,
                                     _load_alias_overrides(alias_path, artifact.agent_index) if config.message.adapter_revision == "legacy-1" else {},
                                     runtime_factory=ArtifactRuntimeFactory(self.settings.onnx_execution_providers))
        if (artifact.format or "onnx") == "onnx":
            session = pipeline.agent.session
            requested = self.settings.onnx_execution_providers
            if any(p not in session.get_providers() for p in requested) or session.get_providers()[0] != requested[0]:
                raise PreparationError("provider_fallback", "ONNX silently fell back from the requested providers", "/runtime/providers")
            inputs, outputs = session.get_inputs(), session.get_outputs()
            for tensors, dimension, path in ((inputs, width, "/runtime/inputs"), (outputs, len(names), "/runtime/outputs")):
                if len(tensors) != 1 or tensors[0].type != "tensor(float)" or len(tensors[0].shape) != 2 or tensors[0].shape[1] != dimension or tensors[0].shape[0] not in (1, None, "batch", "batch_size"):
                    raise PreparationError("onnx_shape_mismatch", "Expected one float32 tensor shaped [1 or batch, declared width]", path)
        # Construct the same record shape used by activation, but do not publish it.
        # Release native sessions after validation; activation must revalidate/reload.
        return PipelineRecord({artifact.agent_index: pipeline}, snapshot.destination / snapshot.reference.manifest_file,
                              snapshot.destination, artifact.agent_index, [artifact.agent_index], alias_path)

    def build_execution(self, candidate_id, destination, source_directory=None):
        """Revalidate copied bytes and compile a reusable execution; caller owns admission."""
        from app.adapters.channel_message import ChannelMessageAdapter
        from app.adapters.execution_context import compile_context
        from app.application.evaluation import CandidateExecution
        from app.core.errors import ExecutionError

        candidate = self.candidate(candidate_id)
        if candidate["state"] != "ready":
            raise PreparationError("candidate_not_ready", "Only ready candidates can be executed", "/candidate_id", 409)
        config = self.configuration(candidate["configuration_revision"])
        if config.message.adapter_revision != "channels-1":
            raise PreparationError("evaluation_adapter_unsupported", "Execution requires an explicit channels-1 configuration", "/message/adapter_revision", 409)
        request = PrepareCandidate.parse_raw(self.store.operation(candidate_id)["request"])
        reference = LocalBundle(directory=source_directory or f"candidates/{candidate_id}/files",
                                manifest_file=request.bundle.manifest_file, alias_file=request.bundle.alias_file,
                                file_hashes=candidate["file_hashes"])
        snapshot = BundleSnapshot(self.store.root, reference, destination, self.settings.max_candidate_bytes)
        manifest = snapshot.collect(config.bundle.manifest_sha256)
        if context_issues(config):
            raise PreparationError("candidate_incompatible", "Candidate no longer meets current capabilities", "/candidate_id", 409)
        artifact, width, names = check_manifest(config, manifest, snapshot)
        try:
            record = self._construct(config, manifest, artifact, width, names, snapshot)
        except PreparationError:
            raise
        except Exception as exc:
            raise ExecutionError("runtime_construction_failed", "Candidate runtime could not be constructed", "/runtime") from exc
        context = compile_context(config, manifest, candidate_id)
        return CandidateExecution(context, ChannelMessageAdapter(context), record.pipeline.agent)

    def evaluate(self, candidate_id, features):
        """Evaluate independently under the bounded builder admission gate."""
        with self.work(), TemporaryDirectory(prefix="evaluation-", dir=self.store.root) as temp:
            execution = self.build_execution(candidate_id, Path(temp) / "files")
            result = execution.evaluate(features)
            result["providers"] = list(execution.runtime.providers)
            return result

    def initialize_files(self):
        if self.settings.configuration_file:
            value = read_json(self.settings.configuration_file.read_bytes(), "/configuration_file")
            config = parse_configuration(value)
            self.put(config.revision, config)
        if self.settings.preparation_request_file:
            value = read_json(self.settings.preparation_request_file.read_bytes(), "/preparation_request_file")
            request = PrepareCandidate.parse_obj(value)
            import hashlib
            from app.preparation.storage import canonical
            key = "startup:" + hashlib.sha256(canonical(value).encode()).hexdigest()
            operation_id = self.submit(key, request)
            self.prepare(operation_id)
