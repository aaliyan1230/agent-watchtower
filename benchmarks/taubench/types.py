# Copyright Sierra (MIT), vendored from sierra-research/tau-bench.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union


@dataclass
class Action:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    annotator: str
    user_id: str
    instruction: str
    actions: List[Action]
    outputs: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Task":
        return cls(
            annotator=d.get("annotator", "unknown"),
            user_id=d.get("user_id", ""),
            instruction=d["instruction"],
            actions=[Action(name=a["name"], kwargs=a.get("arguments", {})) for a in d["actions"]],
            outputs=d.get("outputs", []),
        )


ToHashable = Union[
    str, int, float, Dict[str, "ToHashable"], List["ToHashable"]
]


def to_hashable(item: ToHashable):
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    if isinstance(item, list):
        return tuple(to_hashable(element) for element in item)
    return item


def consistent_hash(value) -> str:
    from hashlib import sha256

    return sha256(str(value).encode("utf-8")).hexdigest()
