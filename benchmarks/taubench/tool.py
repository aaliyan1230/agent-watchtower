# Copyright Sierra (MIT), vendored from sierra-research/tau-bench.
import abc
from typing import Any


class Tool(abc.ABC):
    @staticmethod
    def invoke(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_info() -> dict[str, Any]:
        raise NotImplementedError
