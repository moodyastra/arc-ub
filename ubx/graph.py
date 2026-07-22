"""Exact transition memory; real observations always outrank model predictions."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np


def state_hash(grid: np.ndarray) -> str:
    return hashlib.blake2s(np.asarray(grid, dtype=np.uint8).tobytes(), digest_size=12).hexdigest()


@dataclass(frozen=True)
class Transition:
    source: str
    action_key: str
    target: str
    changed_pixels: int
    terminal: bool
    reset: bool


class TransitionGraph:
    def __init__(self) -> None:
        self.frames: dict[str, np.ndarray] = {}
        self.edges: dict[str, dict[str, Transition]] = defaultdict(dict)
        self.reverse: dict[str, set[str]] = defaultdict(set)
        self.visits: dict[str, int] = defaultdict(int)

    def remember(self, grid: np.ndarray) -> str:
        key = state_hash(grid)
        self.frames.setdefault(key, np.asarray(grid, dtype=np.uint8).copy())
        self.visits[key] += 1
        return key

    def add(self, before: np.ndarray, action_key: str, after: np.ndarray, *, terminal: bool, reset: bool) -> Transition:
        source = self.remember(before)
        target = self.remember(after)
        transition = Transition(
            source, action_key, target,
            int(np.count_nonzero(np.asarray(before) != np.asarray(after))),
            terminal, reset,
        )
        self.edges[source][action_key] = transition
        self.reverse[target].add(source)
        return transition

    def dead_actions(self, state_id: str) -> set[str]:
        return {name for name, edge in self.edges.get(state_id, {}).items() if edge.target == state_id}

    def frontier_actions(self, state_id: str, legal_actions: Iterable[str]) -> list[str]:
        known = self.edges.get(state_id, {})
        dead = self.dead_actions(state_id)
        return [action for action in legal_actions if action not in known and action not in dead]

    def shortest_path(self, start: str, targets: set[str]) -> list[str] | None:
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            node, path = queue.popleft()
            if node in targets:
                return path
            for action, edge in self.edges.get(node, {}).items():
                if edge.target not in seen and not edge.reset:
                    seen.add(edge.target)
                    queue.append((edge.target, path + [action]))
        return None

    def path_to_frontier(self, start: str, legal_actions: Iterable[str]) -> list[str] | None:
        """Return a shortest known-safe route followed by one unexplored action."""
        legal = tuple(action for action in legal_actions if action != "ACTION6")
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            node, path = queue.popleft()
            frontier = self.frontier_actions(node, legal)
            if frontier:
                return path + [frontier[0]]
            for action, edge in self.edges.get(node, {}).items():
                if edge.target not in seen and not edge.reset and not edge.terminal:
                    seen.add(edge.target)
                    queue.append((edge.target, path + [action]))
        return None
