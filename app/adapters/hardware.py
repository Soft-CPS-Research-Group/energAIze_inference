"""Optional runtime hardware diagnostics, separate from inference execution."""

from functools import lru_cache

from app.logging import get_logger


@lru_cache(maxsize=1)
def _cuda_available() -> bool:
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return False
    try:
        import onnx
        from onnx import TensorProto, helper
    except Exception:
        get_logger().warning("onnx.unavailable_for_gpu_check")
        return False

    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "gpu_check",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, producer_name="gpu_check")
    try:
        session = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CUDAExecutionProvider"],
        )
        return "CUDAExecutionProvider" in session.get_providers()
    except Exception:
        return False


