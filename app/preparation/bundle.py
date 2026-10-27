"""Copy validated ordinary files into an isolated candidate before constructing engines."""

import hashlib
import json
import os
from pathlib import Path
import stat

from app.preparation.models import relative_path
from app.preparation.storage import PreparationError
from app.utils.manifest import Manifest


def read_json(data, path):
    def nonfinite(value):
        raise ValueError("Nonfinite JSON number")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(data, parse_constant=nonfinite, object_pairs_hook=pairs)
        if not isinstance(value, dict):
            raise ValueError("Expected JSON object")
        return value
    except (ValueError, UnicodeError) as exc:
        raise PreparationError("invalid_json", str(exc), path) from exc


class BundleSnapshot:
    def __init__(self, allowed_root, reference, destination, max_bytes=512 * 1024 * 1024):
        self.reference = reference
        self.root = Path(allowed_root).resolve()
        self.destination = destination
        self.max_bytes = max_bytes
        self.total = 0
        self.hashes = {}
        self.destination.mkdir(parents=True, mode=0o700)

    def copy(self, name):
        if isinstance(name, str) and name in self.hashes:
            return (self.destination / name).read_bytes()
        try:
            relative_path(name)
            combined = Path(self.reference.directory) / name
            fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                parts = combined.parts
                for component in parts[:-1]:
                    child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    os.close(fd)
                    fd = child
                file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                with os.fdopen(file_fd, "rb") as handle:
                    if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                        raise ValueError("Only ordinary bundle files are supported")
                    data = handle.read(self.max_bytes - self.total + 1)
            finally:
                os.close(fd)
        except (OSError, ValueError, IndexError) as exc:
            raise PreparationError("invalid_bundle_path", f"Cannot read approved bundle file: {name}", f"/bundle/files/{name}") from exc
        self.total += len(data)
        if self.total > self.max_bytes:
            raise PreparationError("bundle_too_large", "Candidate exceeds the configured byte limit", "/bundle")
        digest = hashlib.sha256(data).hexdigest()
        expected = self.reference.file_hashes.get(name)
        if expected is not None and expected != digest:
            raise PreparationError("hash_mismatch", "Declared file hash does not match bytes", f"/bundle/file_hashes/{name}")
        self.hashes[name] = digest
        target = self.destination / name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.write_bytes(data)
        target.chmod(0o400)
        return data

    def collect(self, expected_manifest_hash):
        name = self.reference.manifest_file
        payload = read_json(self.copy(name), "/bundle/manifest")
        if expected_manifest_hash is not None and self.hashes[name] != expected_manifest_hash:
            raise PreparationError("manifest_hash_mismatch", "Manifest differs from the configuration hash", "/bundle/manifest_sha256")
        if type(payload.get("manifest_version")) is not int or payload["manifest_version"] != 1:
            raise PreparationError("manifest_version_unsupported", "Only explicit manifest_version 1 is supported", "/manifest/manifest_version")
        manifest = Manifest.parse_obj(payload)
        alias = self.reference.alias_file or manifest.metadata.get("alias_mapping_path")
        self.alias = alias
        if alias:
            read_json(self.copy(alias), "/bundle/aliases")
        for artifact in manifest.agent.artifacts:
            data = self.copy(artifact.path)
            if artifact.config.get("config_path"):
                read_json(self.copy(artifact.config["config_path"]), "/bundle/config_path")
            if artifact.format in (None, "onnx"):
                import onnx
                try:
                    model = onnx.load_model_from_string(data)
                except Exception as exc:
                    raise PreparationError("invalid_onnx", "Cannot parse ONNX artifact", f"/bundle/files/{artifact.path}") from exc

                def tensors(message):
                    if isinstance(message, onnx.TensorProto):
                        yield message
                    for descriptor, value in message.ListFields():
                        if descriptor.type == descriptor.TYPE_MESSAGE:
                            repeated = descriptor.is_repeated if hasattr(descriptor, "is_repeated") else descriptor.label == descriptor.LABEL_REPEATED
                            children = value if repeated else [value]
                            for child in children:
                                yield from tensors(child)

                for tensor in tensors(model):
                    if tensor.data_location == onnx.TensorProto.EXTERNAL:
                        external = {entry.key: entry.value for entry in tensor.external_data}
                        if len(external) != len(tensor.external_data) or set(external) - {"location", "offset", "length", "checksum"}:
                            raise PreparationError("invalid_external_data", "External ONNX data has duplicate or unsupported addressing metadata", "/bundle/external_data")
                        location = external.get("location", "")
                        try:
                            relative_path(location)
                        except ValueError as exc:
                            raise PreparationError("invalid_external_data", "External ONNX data must stay inside the bundle", "/bundle/external_data") from exc
                        self.copy(str(Path(artifact.path).parent / location))
        for name in self.reference.file_hashes:
            self.copy(name)
        return manifest
