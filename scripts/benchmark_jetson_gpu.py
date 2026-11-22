"""Bounded, synthetic ONNX CPU/TensorRT comparison; never sends device commands.

Generate fixtures with --prepare DIR (Python with onnx), run --cpu DIR inside the
published CPU image, then --gpu DIR on a Jetson with TensorRT 8 and CUDA 11.
The directory is disposable. JSON stdout reports end-to-end single-cycle latency
(including GPU copies/synchronization), numerical parity and the actual backend.
This does not configure the inference service to use TensorRT.
"""
import argparse
import ctypes
import json
from pathlib import Path
import platform
import time

import numpy as np


def prepare(root):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(20261122)
    inputs = rng.uniform(-1, 1, (8, 1, 64)).astype(np.float32)
    np.save(root / "inputs.npy", inputs)
    for name, widths in [("small", [64, 32, 2]), ("dense", [64, 1024, 1024, 1024, 2]),
                          ("large", [64, 2048, 2048, 2048, 2])]:
        nodes, weights, previous = [], [], "x"
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            w = (rng.normal(size=(a, b)) / np.sqrt(a)).astype(np.float32)
            bias = np.full((b,), .1, dtype=np.float32)
            weights += [numpy_helper.from_array(w, "w%d" % i),
                        numpy_helper.from_array(bias, "b%d" % i)]
            nodes += [helper.make_node("MatMul", [previous, "w%d" % i], ["m%d" % i]),
                      helper.make_node("Add", ["m%d" % i, "b%d" % i], ["a%d" % i])]
            previous = "y" if i == len(widths) - 2 else "h%d" % i
            nodes.append(helper.make_node("Relu", ["a%d" % i], [previous]))
        graph = helper.make_graph(nodes, name,
                                  [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])], weights)
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        onnx.save(model, root / (name + ".onnx"))


def measure(run, inputs):
    outputs = np.stack([run(x) for x in inputs])
    for _ in range(10):
        run(inputs[0])
    samples = []
    for i in range(100):
        start = time.perf_counter()
        run(inputs[i % len(inputs)])
        samples.append((time.perf_counter() - start) * 1000)
    return outputs, {"samples": len(samples), "median_ms": float(np.median(samples)),
                     "p95_ms": float(np.percentile(samples, 95))}


def cpu(root):
    import onnxruntime as ort
    assert platform.machine() == "aarch64", "Run this comparison on the actual Jetson"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    results = {}
    for name in ("small", "dense", "large"):
        session = ort.InferenceSession(str(root / (name + ".onnx")), options,
                                       providers=["CPUExecutionProvider"])
        outputs, results[name] = measure(lambda x: session.run(None, {"x": x})[0],
                                        np.load(root / "inputs.npy"))
        np.save(root / (name + "-reference.npy"), outputs)
        results[name]["providers"] = session.get_providers()
    print(json.dumps({"backend": "ONNX Runtime CPU", "version": ort.__version__,
                      "architecture": platform.machine(), "threads": 2, "models": results}))


def gpu(root, fp16=False, models=("small", "dense", "large")):
    import tensorrt as trt
    assert platform.machine() == "aarch64"
    cuda = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
    signatures = {
        "cudaMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        "cudaFree": [ctypes.c_void_p],
        "cudaMemcpy": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
        "cudaDeviceSynchronize": [],
        "cudaGetDevice": [ctypes.POINTER(ctypes.c_int)],
    }
    for name, args in signatures.items():
        fn = getattr(cuda, name)
        fn.argtypes, fn.restype = args, ctypes.c_int

    def call(name, *args):
        code = getattr(cuda, name)(*args)
        if code:
            raise RuntimeError("%s failed with CUDA error %s" % (name, code))

    device = ctypes.c_int()
    call("cudaGetDevice", ctypes.byref(device))
    results = {}
    logger = trt.Logger(trt.Logger.WARNING)
    for name in models:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse((root / (name + ".onnx")).read_bytes()):
            raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * 1024 ** 2)
        # FP32 parity test: do not implicitly enable FP16 or reduced-precision TF32.
        config.clear_flag(trt.BuilderFlag.TF32)
        if fp16:
            assert builder.platform_has_fast_fp16, "FP16 acceleration unavailable"
            config.set_flag(trt.BuilderFlag.FP16)
        serial = builder.build_serialized_network(network, config)
        assert serial is not None, "TensorRT build failed"
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serial)
        context = engine.create_execution_context()
        pointers, buffers = [], []
        try:
            for i in range(engine.num_bindings):
                assert engine.get_binding_dtype(i) == trt.float32
                host = np.empty(tuple(engine.get_binding_shape(i)), dtype=np.float32)
                ptr = ctypes.c_void_p()
                call("cudaMalloc", ctypes.byref(ptr), host.nbytes)
                pointers.append(ptr)
                buffers.append(host)
            xi, yi = engine.get_binding_index("x"), engine.get_binding_index("y")
            assert xi >= 0 and yi >= 0

            def run(x):
                call("cudaMemcpy", pointers[xi], x.ctypes.data, x.nbytes, 1)
                if not context.execute_v2([p.value for p in pointers]):
                    raise RuntimeError("TensorRT execute_v2 failed")
                call("cudaDeviceSynchronize")
                out = buffers[yi]
                call("cudaMemcpy", out.ctypes.data, pointers[yi], out.nbytes, 2)
                return out.copy()

            output, result = measure(run, np.load(root / "inputs.npy"))
            ref = np.load(root / (name + "-reference.npy"))
            np.testing.assert_allclose(output, ref, rtol=1e-2 if fp16 else 3e-4,
                                       atol=2e-3 if fp16 else 1e-5)
            result["max_abs_error"] = float(np.max(np.abs(output - ref)))
            result["parity"] = "passed"
            results[name] = result
        finally:
            for ptr in pointers:
                call("cudaFree", ptr)
    print(json.dumps({"backend": "TensorRT GPU (no CPU fallback)", "version": trt.__version__,
                      "architecture": platform.machine(), "device": device.value,
                      "precision": "FP16 allowed" if fp16 else "FP32", "models": results}))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    mode = args.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", type=Path)
    mode.add_argument("--cpu", type=Path)
    mode.add_argument("--gpu", type=Path)
    args.add_argument("--fp16", action="store_true", help="Allow FP16 GPU tactics; verify with looser tolerance")
    args.add_argument("--models", nargs="+", choices=("small", "dense", "large"),
                      default=["small", "dense", "large"], help="GPU models to run")
    parsed = args.parse_args()
    if parsed.prepare:
        prepare(parsed.prepare)
    elif parsed.cpu:
        cpu(parsed.cpu)
    else:
        gpu(parsed.gpu, parsed.fp16, parsed.models)
