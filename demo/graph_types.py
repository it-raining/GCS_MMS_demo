"""
graph_types.py - Validated node identifiers for the region graph.

The runtime graph still uses string labels for NetworkX compatibility and for
stable result serialization, but all parsing and construction goes through this
module so invalid labels fail early.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Union


SOURCE: Final[str] = "source"
TARGET: Final[str] = "target"


@dataclass(frozen=True)
class RegionNodeId:
    """Typed region node identifier."""

    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"Region index must be nonnegative, got {self.index}")

    @property
    def label(self) -> str:
        return f"R{self.index}"


@dataclass(frozen=True)
class SourceNode:
    """Typed source sentinel."""

    @property
    def label(self) -> str:
        return SOURCE


@dataclass(frozen=True)
class TargetNode:
    """Typed target sentinel."""

    @property
    def label(self) -> str:
        return TARGET


NodeId = Union[RegionNodeId, SourceNode, TargetNode]


def region_node_label(index: int) -> str:
    """Return the canonical string label for a region node."""
    return RegionNodeId(index).label


def is_region_node_id(node_id: str) -> bool:
    """Return True if `node_id` is a valid region node label."""
    if not isinstance(node_id, str) or len(node_id) < 2 or node_id[0] != "R":
        return False
    return node_id[1:].isdigit()


def is_terminal_node_id(node_id: str) -> bool:
    """Return True if `node_id` is source or target."""
    return node_id in (SOURCE, TARGET)


def region_index_from_node_id(node_id: str) -> int:
    """Parse a region index from a validated region node label."""
    if not is_region_node_id(node_id):
        raise ValueError(f"Expected region node label like 'R0', got {node_id!r}")
    return int(node_id[1:])
