"""Resolve a bundle's execution device without silently changing its requirement."""

TENSORRT_PROVIDER = "TensorRTNative"


def requested_providers(config, defaults):
    device = config.get("execution_device")
    if "execution_device" not in config:
        return list(defaults)
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "gpu":
        return [TENSORRT_PROVIDER]
    raise ValueError("execution_device must be cpu or gpu")


def ensure_available(providers, tensorrt_enabled=False):
    if providers == [TENSORRT_PROVIDER]:
        if not tensorrt_enabled:
            raise ValueError("Native TensorRT is not enabled on this deployment")
        from app.adapters.tensorrt_session import TensorRTSession
        TensorRTSession.probe()
    else:
        import onnxruntime as ort
        if not providers or any(p not in ort.get_available_providers() for p in providers):
            raise ValueError("A requested ONNX provider is unavailable")
