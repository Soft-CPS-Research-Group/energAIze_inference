"""Execution failures independent of HTTP and engine libraries."""


class InferenceError(RuntimeError):
    def __init__(self, code: str, detail: str, path: str = "/"):
        super().__init__(detail)
        self.code, self.detail, self.path = code, detail, path

    def document(self):
        return {"code": self.code, "path": self.path, "detail": self.detail}


class InputError(InferenceError):
    """A consumed input is missing, ambiguous or invalid."""


class ExecutionError(InferenceError):
    """The engine failed or produced an invalid result; never a valid decision."""
