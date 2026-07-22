"""Compatibility adapter between UB-X engines and the existing worker loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ub import GameCommand, GridPosition, LunaDecision, ObjectObservation, PlannerResult

from .engine import GraphBaselineEngine, OfflineUBXEngine
from .schemas import ActionOutcome, Observation, SearchBudget


class OfflinePlannerAdapter:
    needs_exact_grid = True

    def __init__(
        self,
        rules_path: Path,
        *,
        planner: str,
        model_path: Path | None,
        search_budget_ms: int,
        max_hypotheses: int,
        disabled_experts: tuple[str, ...] = (),
    ) -> None:
        self.rules = json.loads(rules_path.read_text(encoding="utf-8"))
        self.model = planner
        self.thread_id = "offline"
        self.turn = 0
        self.search_budget_ms = max(1, search_budget_ms)
        self.engine = (
            GraphBaselineEngine(max_hypotheses=max_hypotheses)
            if planner == "graph_baseline"
            else OfflineUBXEngine(model_path, max_hypotheses=max_hypotheses, disabled_experts=disabled_experts)
        )
        self._last_observation: Observation | None = None

    def reset_thread(self) -> None:
        self.thread_id = "offline"
        self._last_observation = None

    def decide(self, screenshots: list[tuple[str, Path]], context: dict[str, object]) -> PlannerResult:
        del screenshots
        grid = np.asarray(context["exact_grid"], dtype=np.uint8)
        observation = Observation(
            grid=grid,
            previous_grid=self._last_observation.grid if self._last_observation else None,
            action_space=tuple(str(value) for value in context["available_actions"]),
            level=int(context["current_level"]),
            action_number=int(context["actions_taken_in_current_level"]),
            metadata={"state": context["environment_state"]},
        )
        belief = self.engine.observe(observation)
        allowed = int(context["maximum_commands_this_turn"])
        plan = self.engine.plan(
            belief,
            SearchBudget(milliseconds=self.search_budget_ms, hypothesis_width=len(belief.hypotheses), beam_width=max(allowed, 8)),
        )
        plan.actions = plan.actions[:allowed]
        self._last_observation = observation
        self.turn += 1
        objects = [self._object(entity) for entity in belief.entities if not self._background_like(entity)][:12]
        if not objects:
            objects = [ObjectObservation(
                memory_id=None,
                scene_candidate_ids=[],
                position=GridPosition(x_min=0, x_max=63, y_min=0, y_max=63),
                color="indexed",
                shapes=["unresolved scene"],
                real_life_possible_object="unknown mechanism",
                importance=1,
                speculations=["requires information-gain probe", "may encode a non-object rule"],
                status="unknown",
            )]
        confidence = float(plan.confidence)
        commands = [
            GameCommand(
                action=action["action"], x=action.get("x"), y=action.get("y"),
                check_after=index == len(plan.actions) - 1,
            )
            for index, action in enumerate(plan.actions)
        ]
        decision = LunaDecision(
            differences=self._differences(observation),
            similarities=self._similarities(belief),
            objects=objects,
            revised_hypotheses=[f"{item.hypothesis_id}:{item.confidence:.2f}" for item in belief.hypotheses[:8]],
            commands=commands,
            batch_reason="deliberate information probe" if len(commands) == 1 else "verified macro to next decision point",
            expected_change="; ".join(str(effect) for effect in plan.expected_effects[:2]),
            goal_status=self._goal_status(plan.goal),
            confidence=confidence,
            scene_coverage=min(1.0, len(belief.entities) / 12.0),
            level_model_confidence=confidence,
            stuck=False,
            request_bfs=False,
            bfs_reason="exact graph search is internal and only used after model/search exhaustion",
        )
        return PlannerResult(decision=decision, thread_id=self.thread_id, usage={"offline": True, **plan.diagnostics})

    def decide_fresh(self, screenshots: list[tuple[str, Path]], context: dict[str, object]) -> PlannerResult:
        return self.decide(screenshots, context)

    def acknowledge_transition(
        self,
        before: np.ndarray,
        after: np.ndarray,
        command: GameCommand,
        *,
        action_number: int,
        level: int,
        available_actions: list[int],
        terminal: bool,
        reset: bool,
    ) -> None:
        names = tuple(f"ACTION{value}" for value in available_actions)
        before_observation = self._last_observation or Observation(np.asarray(before), None, names, level, max(0, action_number - 1))
        after_observation = Observation(np.asarray(after), np.asarray(before), names, level, action_number)
        self.engine.acknowledge(ActionOutcome(
            action=command.model_dump(mode="json"), before=before_observation, after=after_observation,
            terminal=terminal, reset=reset,
        ))
        self._last_observation = after_observation

    @staticmethod
    def _background_like(entity: Any) -> bool:
        return entity.area > 1024 and not entity.is_ui

    @staticmethod
    def _object(entity: Any) -> ObjectObservation:
        x0, y0, x1, y1 = entity.bbox
        role = max(entity.role_probabilities, key=entity.role_probabilities.get)
        return ObjectObservation(
            memory_id=None,
            scene_candidate_ids=[entity.entity_id],
            position=GridPosition(x_min=x0, x_max=x1, y_min=y0, y_max=y1),
            color="/".join(str(value) for value in entity.colors),
            shapes=[entity.shape, *(f"{name} symmetry" for name in entity.symmetry[:2])],
            real_life_possible_object=role,
            importance=max(1, min(10, int(10 * (1.0 - entity.role_probabilities.get("unknown", 0.0))))),
            speculations=[f"may be {role}", "may participate in a geometric or temporal rule"],
            status="goal" if role == "goal" else "visible",
        )

    @staticmethod
    def _differences(observation: Observation) -> list[str]:
        if observation.previous_grid is None:
            return ["initial exact grid"]
        changed = int(np.count_nonzero(observation.grid != observation.previous_grid))
        return [f"{changed} indexed cells changed"]

    @staticmethod
    def _similarities(belief: Any) -> list[str]:
        pairs = belief.features["scene"].get("perceptual_summary", {}).get("geometry_matches", [])
        return [str(pair)[:220] for pair in pairs[:8]] or ["no verified geometric pair yet"]

    @staticmethod
    def _goal_status(goal: str) -> str:
        if "collect" in goal:
            return "collecting"
        if "reach" in goal or "match" in goal:
            return "approaching_goal"
        return "exploring"
