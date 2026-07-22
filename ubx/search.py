"""Budgeted candidate generation and policy-guided verified beam planning."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np

from .schemas import ActionPlan, BeliefState, SearchBudget


@dataclass(frozen=True)
class Candidate:
    action: dict[str, Any]
    score: float
    expected: dict[str, Any]


def action_key(action: dict[str, Any]) -> str:
    if action.get("action") == "ACTION6":
        return f"ACTION6:{int(action['x'])},{int(action['y'])}"
    return str(action["action"])


def action_from_key(key: str) -> dict[str, Any]:
    if key.startswith("ACTION6:"):
        x, y = key.split(":", 1)[1].split(",", 1)
        return {"action": "ACTION6", "x": int(x), "y": int(y)}
    return {"action": key, "x": None, "y": None}


def generate_candidates(belief: BeliefState) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for action in belief.legal_actions:
        if action != "ACTION6":
            candidates.append({"action": action, "x": None, "y": None})
    if "ACTION6" not in belief.legal_actions:
        return candidates
    points: set[tuple[int, int]] = set()
    for entity in belief.entities:
        x0, y0, x1, y1 = entity.bbox
        cx, cy = int(round(entity.centroid[0])), int(round(entity.centroid[1]))
        points.update({(cx, cy), (x0, y0), (x1, y1), ((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1), (x0, (y0 + y1) // 2), (x1, (y0 + y1) // 2)})
        points.add((63 - cx, cy))
        points.add((cx, 63 - cy))
    points.update((x, y) for y in range(4, 64, 8) for x in range(4, 64, 8))
    candidates.extend({"action": "ACTION6", "x": x, "y": y} for x, y in sorted(points) if 0 <= x < 64 and 0 <= y < 64)
    return candidates


class WorldModelSearch:
    def __init__(self, scorer: Callable[[BeliefState, dict[str, Any]], tuple[float, dict[str, Any]]] | None = None) -> None:
        self.scorer = scorer

    def plan(self, belief: BeliefState, budget: SearchBudget) -> ActionPlan:
        started = time.perf_counter()
        if belief.observation.action_number >= 3:
            terminal_targets = {
                edge.target
                for edges in belief.transition_graph.edges.values()
                for edge in edges.values()
                if edge.terminal
            }
            exact_path = belief.transition_graph.shortest_path(belief.state_id, terminal_targets) if terminal_targets else None
            source = "exact_terminal_path"
            if exact_path is None:
                exact_path = belief.transition_graph.path_to_frontier(belief.state_id, belief.legal_actions)
                source = "exact_frontier_path"
            if exact_path and (source == "exact_terminal_path" or len(exact_path) >= 2):
                keys = exact_path[:8]
                return ActionPlan(
                    actions=[action_from_key(key) for key in keys],
                    expected_effects=[{"exact_graph": index < len(keys) - 1, "changed": None} for index, _ in enumerate(keys)],
                    interruption_conditions=("prediction_diverged", "reset", "terminal", "resource_changed", "level_changed"),
                    confidence=0.95 if source == "exact_terminal_path" else 0.78,
                    goal="reach_verified_terminal" if source == "exact_terminal_path" else "explore_nearest_frontier",
                    source=source,
                    diagnostics={"known_path_length": len(exact_path), "elapsed_ms": (time.perf_counter() - started) * 1000},
                )
        dead = belief.transition_graph.dead_actions(belief.state_id)
        candidates: list[Candidate] = []
        for action in generate_candidates(belief):
            if (time.perf_counter() - started) * 1000 >= budget.milliseconds:
                break
            key = action_key(action)
            if key in dead:
                continue
            score, expected = self._score(belief, action)
            candidates.append(Candidate(action, score, expected))
        candidates.sort(key=lambda item: (-item.score, action_key(item.action)))
        if not candidates:
            fallback = {"action": belief.legal_actions[0], "x": None, "y": None}
            candidates = [Candidate(fallback, -1.0, {"changed": False})]

        deliberate = belief.observation.action_number < 3 or belief.uncertainty > 0.72
        length = 1 if deliberate else min(8, max(3, int((1.0 - belief.uncertainty) * 9)))
        top = candidates[: max(1, min(budget.beam_width, len(candidates)))]
        actions: list[dict[str, Any]] = []
        effects: list[dict[str, Any]] = []
        for step in range(length):
            candidate = top[step % len(top)]
            actions.append(dict(candidate.action))
            effects.append(dict(candidate.expected))
        confidence = float(np.clip(1.0 - belief.uncertainty, 0.05, 0.98))
        return ActionPlan(
            actions=actions,
            expected_effects=effects,
            interruption_conditions=("prediction_diverged", "reset", "terminal", "resource_changed", "level_changed"),
            confidence=confidence,
            goal=max(belief.goals, key=belief.goals.get, default="exploring"),
            source="policy_guided_beam" if self.scorer else "graph_information_gain",
            diagnostics={"candidate_count": len(candidates), "dead_actions": sorted(dead), "elapsed_ms": (time.perf_counter() - started) * 1000},
        )

    def _score(self, belief: BeliefState, action: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        if self.scorer is not None:
            return self.scorer(belief, action)
        key = action_key(action)
        known = belief.transition_graph.edges.get(belief.state_id, {}).get(key)
        novelty = 1.0 if known is None else 0.05 / max(1, belief.transition_graph.visits[known.target])
        click_bonus = 0.0
        if action["action"] == "ACTION6":
            x, y = int(action["x"]), int(action["y"])
            for entity in belief.entities:
                if entity.bbox[0] <= x <= entity.bbox[2] and entity.bbox[1] <= y <= entity.bbox[3]:
                    click_bonus = max(click_bonus, 0.4 + 0.5 * (1.0 - entity.role_probabilities.get("unknown", 0.0)))
        risk = 0.7 if known and known.reset else 0.0
        return novelty + click_bonus - risk, {"novelty": novelty, "risk": risk, "changed": known.changed_pixels > 0 if known else None}
