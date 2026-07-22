"""Exact-grid, difference, object, geometry, UI, and trajectory perception."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ub_scene import SceneAnalyzer

from .schemas import EntitySlot, Observation


class MultiviewPerception:
    def __init__(self, max_slots: int = 64, edge_width: int = 6) -> None:
        self.max_slots = max_slots
        self.edge_width = edge_width
        self.analyzer = SceneAnalyzer()
        self.trajectories: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
        self.level: int | None = None

    def encode(self, observation: Observation) -> tuple[list[EntitySlot], dict[str, Any]]:
        if self.level != observation.level:
            self.analyzer.reset_level()
            self.trajectories.clear()
            self.level = observation.level
        snapshot = self.analyzer.analyze(observation.grid, observation.action_number)
        components = sorted(snapshot["components"], key=self._salience, reverse=True)
        entities: list[EntitySlot] = []
        for component in components[: self.max_slots]:
            centroid = tuple(float(v) for v in component["centroid"])
            track = self.trajectories[component["id"]]
            track.append((observation.action_number, centroid[0], centroid[1]))
            del track[:-8]
            bbox = tuple(int(v) for v in component["bbox"])
            is_ui = int(component["area"]) <= 1024 and (
                bbox[0] < self.edge_width or bbox[1] < self.edge_width
                or bbox[2] >= 64 - self.edge_width or bbox[3] >= 64 - self.edge_width
            )
            symmetry = tuple(name for name, active in component["symmetry"].items() if active)
            entities.append(EntitySlot(
                entity_id=component["id"],
                colors=(int(component["color_id"]),),
                bbox=bbox,
                centroid=centroid,
                area=int(component["area"]),
                shape=str(component["shape_kind"]),
                relations={k: tuple(v) for k, v in component["relations"].items() if isinstance(v, list)},
                trajectory=list(track),
                role_probabilities=self._roles(component, is_ui),
                symmetry=symmetry,
                is_ui=is_ui,
            ))
        previous = observation.previous_grid
        delta = np.zeros_like(observation.grid, dtype=np.uint8) if previous is None else (previous != observation.grid).astype(np.uint8)
        views = {
            "grid": observation.grid,
            "delta": delta,
            "patch_2x2": self._patch_histogram(observation.grid, 2),
            "patch_4x4": self._patch_histogram(observation.grid, 4),
            "edge_colors": self._edge_colors(observation.grid),
            "scene": snapshot,
        }
        return entities, views

    @staticmethod
    def _salience(component: dict[str, Any]) -> tuple[int, int, int]:
        role = component.get("role_hint", "")
        meaningful = int(role not in {"background_or_field", "wall_or_region", ""})
        changed = int(component.get("change", {}).get("events", ["unchanged"]) != ["unchanged"])
        return meaningful, changed, -int(component["area"])

    @staticmethod
    def _roles(component: dict[str, Any], is_ui: bool) -> dict[str, float]:
        role = str(component.get("role_hint", "unknown"))
        result = {"unknown": 0.35, "player": 0.15, "goal": 0.15, "collectible": 0.15, "hazard": 0.1, "ui": 0.1}
        if is_ui:
            result.update(ui=0.65, unknown=0.15)
        if role and role not in {"unknown", "background_or_field", "wall_or_region"}:
            result[role] = max(result.get(role, 0.0), 0.55)
        total = sum(result.values())
        return {key: value / total for key, value in result.items()}

    @staticmethod
    def _patch_histogram(grid: np.ndarray, size: int) -> np.ndarray:
        h, w = grid.shape
        result = np.zeros((h // size, w // size, 16), dtype=np.float32)
        for color in range(16):
            mask = (grid == color).reshape(h // size, size, w // size, size)
            result[:, :, color] = mask.mean(axis=(1, 3))
        return result

    @staticmethod
    def _edge_colors(grid: np.ndarray) -> np.ndarray:
        edge = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
        return np.bincount(edge, minlength=16).astype(np.int32)
