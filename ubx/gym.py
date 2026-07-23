"""Procedural mechanic-family gym for hidden-first generalization training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


MECHANIC_FAMILIES = (
    "navigation", "collection", "matching", "transformation", "switch",
    "resource", "timer", "avoidance", "click", "mixed", "delayed", "partial",
)


@dataclass(frozen=True)
class GymConfig:
    family: str = "navigation"
    seed: int = 0
    max_steps: int = 96
    stochastic: bool = False


class ProceduralArcEnv:
    """A small exact simulator whose superficial colors/layout never define its rule."""

    def __init__(self, config: GymConfig) -> None:
        if config.family not in MECHANIC_FAMILIES:
            raise ValueError(f"unknown mechanic family: {config.family}")
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.grid = np.zeros((64, 64), dtype=np.uint8)
        self.steps = 0
        self.done = False
        self.pending_effect = 0
        self.resource = 24
        self.player = (0, 0)
        self.goal = (0, 0)
        self.collectible: tuple[int, int] | None = None
        self.switch: tuple[int, int] | None = None
        self.switch_activated = False
        self.hazards: set[tuple[int, int]] = set()
        self.walls: set[tuple[int, int]] = set()
        self.colors: dict[str, int] = {}
        self.reset()

    @property
    def available_actions(self) -> tuple[str, ...]:
        if self.config.family == "click":
            return ("ACTION6",)
        if self.config.family == "mixed":
            return ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6")
        return ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7")

    def reset(self) -> np.ndarray:
        self.steps, self.done, self.pending_effect, self.resource = 0, False, 0, 24
        self.switch_activated = False
        palette = self.rng.choice(np.arange(1, 16), size=6, replace=False)
        self.colors = dict(zip(("floor", "wall", "player", "goal", "item", "ui"), map(int, palette), strict=True))
        self.player = (int(self.rng.integers(8, 56)), int(self.rng.integers(8, 56)))
        self.goal = (int(self.rng.integers(6, 58)), int(self.rng.integers(6, 58)))
        while self.goal == self.player:
            self.goal = (int(self.rng.integers(6, 58)), int(self.rng.integers(6, 58)))
        self.collectible = (int(self.rng.integers(6, 58)), int(self.rng.integers(6, 58))) if self.config.family in {"collection", "matching", "mixed"} else None
        self.switch = (int(self.rng.integers(6, 58)), int(self.rng.integers(6, 58))) if self.config.family in {"switch", "transformation", "delayed", "mixed", "click"} else None
        self.hazards = {(int(self.rng.integers(5, 59)), int(self.rng.integers(5, 59))) for _ in range(5)} if self.config.family in {"avoidance", "mixed"} else set()
        self.walls = set()
        for _ in range(38):
            x, y = int(self.rng.integers(4, 60)), int(self.rng.integers(4, 60))
            if (x, y) not in {self.player, self.goal, self.collectible, self.switch}:
                self.walls.add((x, y))
        self._render()
        return self.grid.copy()

    def step(self, action: dict[str, Any]) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("step called after terminal state")
        self.steps += 1
        before = self.grid.copy()
        name = str(action["action"])
        if name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}:
            dx, dy = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}[name]
            target = (self.player[0] + dx, self.player[1] + dy)
            if 1 <= target[0] <= 62 and 1 <= target[1] <= 62 and target not in self.walls:
                self.player = target
                self.resource -= 1
        elif name == "ACTION6" and self.switch is not None:
            if (int(action.get("x", -1)), int(action.get("y", -1))) == self.switch:
                self.switch_activated = True
                self.pending_effect = 1 if self.config.family != "delayed" else 3
        elif name in {"ACTION5", "ACTION7"} and self.player == self.switch:
            self.switch_activated = True
            self.pending_effect = 1 if self.config.family != "delayed" else 3

        collected = self.collectible is not None and self.player == self.collectible
        if collected:
            self.collectible = None
        if self.pending_effect:
            self.pending_effect -= 1
            if self.pending_effect == 0 and self.switch is not None:
                self.walls = {point for point in self.walls if (point[0] + point[1]) % 2}
        reset = self.player in self.hazards or self.resource <= 0
        requires_item = self.config.family in {"collection", "matching", "mixed"}
        requires_switch = self.config.family in {"switch", "transformation", "delayed", "mixed", "click"}
        switch_satisfied = not requires_switch or (self.switch_activated and self.pending_effect == 0)
        reached_goal = switch_satisfied if self.config.family == "click" else self.player == self.goal
        self.done = reached_goal and (not requires_item or self.collectible is None) and switch_satisfied
        reward = 1.0 if self.done else -0.002
        if reset:
            reward, self.done = -1.0, True
        if collected:
            reward += 0.2
        if self.steps >= self.config.max_steps:
            self.done = True
        self._render()
        return self.grid.copy(), reward, self.done, {"changed_pixels": int(np.count_nonzero(before != self.grid)), "reset": reset, "collected": collected}

    def oracle_action(self) -> dict[str, Any]:
        target = self.collectible or self.switch or self.goal
        if self.config.family == "click" and self.switch is not None:
            return {"action": "ACTION6", "x": self.switch[0], "y": self.switch[1]}
        if self.player == self.switch and self.switch is not None:
            return {"action": "ACTION5", "x": None, "y": None}
        dx, dy = target[0] - self.player[0], target[1] - self.player[1]
        options = []
        if dy < 0: options.append("ACTION1")
        if dy > 0: options.append("ACTION2")
        if dx < 0: options.append("ACTION3")
        if dx > 0: options.append("ACTION4")
        for name in options + list(self.available_actions):
            if name == "ACTION6":
                continue
            mx, my = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}.get(name, (0, 0))
            if (self.player[0] + mx, self.player[1] + my) not in self.walls:
                return {"action": name, "x": None, "y": None}
        return {"action": self.available_actions[0], "x": None, "y": None}

    def _render(self) -> None:
        self.grid.fill(self.colors["floor"])
        self.grid[[0, -1], :] = self.colors["wall"]
        self.grid[:, [0, -1]] = self.colors["wall"]
        for x, y in self.walls: self.grid[y, x] = self.colors["wall"]
        for x, y in self.hazards: self.grid[y, x] = 8
        gx, gy = self.goal
        self.grid[max(1, gy - 1):min(63, gy + 2), max(1, gx - 1):min(63, gx + 2)] = self.colors["goal"]
        if self.collectible is not None:
            x, y = self.collectible
            self.grid[y, x] = self.colors["item"]
        if self.switch is not None:
            x, y = self.switch
            self.grid[y:y + 2, x:x + 2] = self.colors["ui"]
        px, py = self.player
        self.grid[py:py + 2, px:px + 2] = self.colors["player"]
        if self.config.family in {"resource", "timer", "mixed"}:
            self.grid[62, 2:2 + max(0, self.resource)] = self.colors["ui"]


def family_split(family: str) -> str:
    return "heldout" if family in {"collection", "delayed", "mixed"} else "train"
