"""JetPack 5 TensorRT execution behind the small session port used by ONNX.

Engines are built from the verified ONNX snapshot on the target, in FP32. No
serialized engine, plugin code or CPU fallback is loaded from a bundle. CUDA
buffers belong to the session and requests share its execution context under a
lock. Native imports are lazy so CPU images do not require NVIDIA libraries.
"""

import ctypes
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np

from app.adapters.onnx_backend import TENSORRT_PROVIDER


class CudaRuntime:
    def __init__(self):
        self.library = ctypes.CDLL("libcudart.so.11.0")
        for name, args in {
            "cudaSetDevice": [ctypes.c_int],
            "cudaMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
            "cudaFree": [ctypes.c_void_p],
            "cudaMemcpy": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
            "cudaDeviceSynchronize": [],
        }.items():
            fn = getattr(self.library, name)
            fn.argtypes, fn.restype = args, ctypes.c_int
        self.call("cudaSetDevice", 0)

    def call(self, name, *args):
        code = getattr(self.library, name)(*args)
        if code:
            raise RuntimeError(f"{name} failed with CUDA error {code}")


class TensorRTSession:
    @staticmethod
    def probe():
        import tensorrt as trt
        if not trt.__version__.startswith("8.5."):
            raise RuntimeError("Native TensorRT adapter requires TensorRT 8.5")
        CudaRuntime()

    def __init__(self, path):
        import tensorrt as trt
        self._lock = Lock()
        self._pointers = []
        self._context = self._engine = self._runtime = None
        self.probe()
        self._cuda = CudaRuntime()
        self._logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self._logger, "")
        try:
            builder = trt.Builder(self._logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, self._logger)
            # parse_from_file resolves verified external-data files beside the ONNX.
            if not parser.parse_from_file(str(Path(path))):
                raise ValueError("TensorRT ONNX parsing failed: " + "; ".join(
                    str(parser.get_error(i)) for i in range(parser.num_errors)))
            if network.num_inputs != 1 or network.num_outputs != 1:
                raise ValueError("TensorRT requires one input and one output")
            for tensor in (network.get_input(0), network.get_output(0)):
                shape = tuple(tensor.shape)
                if tensor.dtype != trt.float32 or len(shape) != 2 or shape[0] != 1 or shape[1] <= 0:
                    raise ValueError("TensorRT requires fixed float32 tensors [1, width]")
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * 1024 ** 2)
            config.clear_flag(trt.BuilderFlag.TF32)
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("TensorRT engine construction failed")
            self._runtime = trt.Runtime(self._logger)
            self._engine = self._runtime.deserialize_cuda_engine(serialized)
            if self._engine is None:
                raise RuntimeError("TensorRT engine loading failed")
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError("TensorRT execution context creation failed")
            self._inputs, self._outputs = [], []
            for i in range(self._engine.num_bindings):
                if self._engine.get_binding_dtype(i) != trt.float32:
                    raise ValueError("TensorRT bindings must be float32")
                tensor = SimpleNamespace(name=self._engine.get_binding_name(i),
                    shape=list(self._engine.get_binding_shape(i)), type="tensor(float)")
                if self._engine.binding_is_input(i):
                    self._inputs.append(tensor)
                    self._input_index = i
                else:
                    self._outputs.append(tensor)
                    self._output_index = i
                pointer = ctypes.c_void_p()
                self._cuda.call("cudaMalloc", ctypes.byref(pointer), int(np.prod(tensor.shape)) * 4)
                self._pointers.append(pointer)
            self._output = np.empty(self._outputs[0].shape, dtype=np.float32)
            # Exercise actual GPU execution before candidate readiness/activation.
            self.run(None, {self._inputs[0].name: np.zeros(self._inputs[0].shape, dtype=np.float32)})
        except BaseException:
            self.close()
            raise

    def get_providers(self):
        return [TENSORRT_PROVIDER]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feed):
        tensor = self._inputs[0]
        if output_names is not None or set(feed) != {tensor.name}:
            raise ValueError("Expected all outputs and the declared input")
        value = np.asarray(feed[tensor.name])
        if value.dtype != np.float32 or list(value.shape) != tensor.shape:
            raise ValueError("Input must match the declared float32 shape")
        value = np.ascontiguousarray(value)
        with self._lock:
            if not self._pointers:
                raise RuntimeError("TensorRT session is closed")
            # CUDA device selection is thread-local; HTTP workers may differ from
            # the preparation thread. Buffers use the device's primary context.
            self._cuda.call("cudaSetDevice", 0)
            self._cuda.call("cudaMemcpy", self._pointers[self._input_index], value.ctypes.data, value.nbytes, 1)
            if not self._context.execute_v2([p.value for p in self._pointers]):
                raise RuntimeError("TensorRT execution failed")
            self._cuda.call("cudaDeviceSynchronize")
            self._cuda.call("cudaMemcpy", self._output.ctypes.data,
                            self._pointers[self._output_index], self._output.nbytes, 2)
            return [self._output.copy()]

    def close(self):
        with self._lock:
            if hasattr(self, "_cuda"):
                self._cuda.call("cudaSetDevice", 0)
                for pointer in self._pointers:
                    self._cuda.call("cudaFree", pointer)
                self._pointers.clear()
            self._context = self._engine = self._runtime = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Device/process teardown must not mask the original failure.
