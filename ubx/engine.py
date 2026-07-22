"""Interchangeable graph and sparse-neural UB-X decision engines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .graph import TransitionGraph, state_hash
from .hypotheses import HypothesisBank
from .perception import MultiviewPerception
from .schemas import ActionOutcome, ActionPlan, BeliefState, BeliefUpdate, Observation, SearchBudget
from .search import WorldModelSearch, action_key


class GraphBaselineEngine:
    def __init__(self, *, max_hypotheses: int = 8) -> None:
        self.graph = TransitionGraph()
        self.perception = MultiviewPerception()
        self.hypotheses = HypothesisBank(max_hypotheses)
        self.current: BeliefState | None = None
        self.last_plan: ActionPlan | None = None
        self.search = WorldModelSearch()

    def observe(self, observation: Observation) -> BeliefState:
        entities, views = self.perception.encode(observation)
        state_id = self.graph.remember(observation.grid)
        goal_scores = {hypothesis.goal: hypothesis.confidence for hypothesis in self.hypotheses.items}
        resources = {f"edge_color_{index}": float(value) for index, value in enumerate(views["edge_colors"]) if value}
        belief = BeliefState(
            observation=observation,
            entities=entities,
            hypotheses=list(self.hypotheses.items),
            state_id=state_id,
            transition_graph=self.graph,
            goals=goal_scores or {"exploring": 1.0},
            resources=resources,
            uncertainty=self.hypotheses.disagreement(),
            legal_actions=observation.action_space,
            features=views,
        )
        self.current = belief
        return belief

    def plan(self, belief: BeliefState, budget: SearchBudget) -> ActionPlan:
        self.last_plan = self.search.plan(belief, budget)
        return self.last_plan

    def acknowledge(self, outcome: ActionOutcome) -> BeliefUpdate:
        key = action_key(outcome.action)
        transition = self.graph.add(outcome.before.grid, key, outcome.after.grid, terminal=outcome.terminal, reset=outcome.reset)
        entities = self.current.entities if self.current is not None else []
        hypotheses = self.hypotheses.update(outcome, entities)
        predicted_change = None
        if self.last_plan:
            for action, effect in zip(self.last_plan.actions, self.last_plan.expected_effects, strict=False):
                if action_key(action) == key:
                    predicted_change = effect.get("changed")
                    break
        observed_change = transition.changed_pixels > 0
        error = 0.0 if predicted_change is None or predicted_change == observed_change else 1.0
        return BeliefUpdate(
            prediction_error=error,
            meaningful_divergence=error > 0.25 or outcome.reset,
            changed_pixels=transition.changed_pixels,
            hypothesis_confidences={item.hypothesis_id: item.confidence for item in hypotheses},
        )


class OfflineUBXEngine(GraphBaselineEngine):
    def __init__(self, model_path: Path | None = None, *, max_hypotheses: int = 8, disabled_experts: tuple[str, ...] = ()) -> None:
        super().__init__(max_hypotheses=max_hypotheses)
        self.model_path = model_path
        self.disabled_experts = disabled_experts
        self.model: Any = None
        self._torch: Any = None
        self._latest_neural: dict[str, Any] | tuple[Any, ...] | None = None
        if model_path is not None:
            self._load_model(model_path)
            self.search = WorldModelSearch(self._neural_score)

    def _load_model(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            import torch
            from .model import UBXModel, UBXModelConfig
        except ImportError as exc:
            raise RuntimeError("Install requirements-ubx-train.txt to load a neural checkpoint") from exc
        self._torch = torch
        try:
            self.model = torch.jit.load(str(path), map_location="cpu").eval()
        except (RuntimeError, ValueError):
            payload = torch.load(path, map_location="cpu", weights_only=True)
            config = UBXModelConfig(**payload.get("config", {}))
            model = UBXModel(config)
            model.load_state_dict(payload["model"])
            self.model = model.eval()

    def observe(self, observation: Observation) -> BeliefState:
        belief = super().observe(observation)
        if self.model is not None:
            torch = self._torch
            grid = torch.from_numpy(observation.grid).unsqueeze(0).long()
            previous = torch.from_numpy(observation.previous_grid if observation.previous_grid is not None else observation.grid).unsqueeze(0).long()
            with torch.inference_mode():
                try:
                    self._latest_neural = self.model(grid, previous, disabled_experts=self.disabled_experts)
                except (TypeError, RuntimeError):
                    self._latest_neural = self.model(grid, previous)
            uncertainty = self._tensor_output("uncertainty", 4)
            if uncertainty is not None:
                belief.uncertainty = float(uncertainty.float().mean().clamp(0, 1))
            belief.features["neural_enabled"] = True
            belief.features["disabled_experts"] = self.disabled_experts
        else:
            belief.features["neural_enabled"] = False
        return belief

    def plan(self, belief: BeliefState, budget: SearchBudget) -> ActionPlan:
        plan = super().plan(belief, budget)
        drafts = self._tensor_output("next_action_logits", 5)
        if drafts is None or belief.observation.action_number < 3 or plan.source.startswith("exact_"):
            return plan
        legal = set(belief.legal_actions)
        dead = belief.transition_graph.dead_actions(belief.state_id)
        drafted: list[dict[str, Any]] = []
        for horizon in range(min(8, drafts.shape[1])):
            ranking = self._torch.argsort(drafts[0, horizon], descending=True).tolist()
            selected = None
            for index in ranking:
                name = f"ACTION{index}"
                if index and name in legal and name != "ACTION6" and name not in dead:
                    selected = {"action": name, "x": None, "y": None}
                    break
            if selected is not None:
                drafted.append(selected)
        if len(drafted) >= 3:
            plan.actions = drafted
            plan.expected_effects = [{"latent_horizon": index + 1, "changed": None} for index in range(len(drafted))]
            plan.source = "multi_step_sparse_world_model"
        return plan

    def _tensor_output(self, name: str, tuple_index: int) -> Any:
        if isinstance(self._latest_neural, dict):
            return self._latest_neural.get(name)
        if isinstance(self._latest_neural, tuple) and len(self._latest_neural) > tuple_index:
            return self._latest_neural[tuple_index]
        return None

    def _neural_score(self, belief: BeliefState, action: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        action_logits = self._tensor_output("action_logits", 0)
        click_logits = self._tensor_output("click_logits", 1)
        value = self._tensor_output("value", 2)
        events = self._tensor_output("event_logits", 3)
        if action_logits is None:
            return 0.0, {"changed": None}
        index = int(str(action["action"]).removeprefix("ACTION"))
        score = float(action_logits[0, index])
        if index == 6 and click_logits is not None:
            score += float(click_logits[0, int(action["y"]), int(action["x"])])
        completion = float(value[0]) if value is not None else 0.0
        reset_risk = float(self._torch.sigmoid(events[0, 2])) if events is not None else 0.0
        score += completion - 1.5 * reset_risk
        return score, {"changed": None, "completion": completion, "reset_risk": reset_risk}
