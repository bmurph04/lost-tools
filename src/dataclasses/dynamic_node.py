from typing import Any
from dataclasses import dataclass

@dataclass(frozen=True)
class DynamicNode:
    """One live node, paired with the track that owns it."""
    slot: int
    object_id: int
    class_id: int
    mean: Any # (3,)
    