"""Stable, language-free runtime contracts for UB-X."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Observation:
    grid: np.ndarray
    previous_grid: np.ndarray | None
    action_space: tuple[str, ...]
    level: int = 1
    action_number: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid, dtype=np.uint8)
        if grid.shape != (64, 64):
            raise ValueError(f"UB-X requires a 64x64 grid, got {grid.shape}")
        if np.any(grid > 15):
            raise ValueError("ARC indexed colors must be in 0..15")
        object.__setattr__(self, "grid", grid.copy())
        if self.previous_grid is not None:
            previous = np.asarray(self.previous_grid, dtype=np.uint8)
            if previous.shape != grid.shape:
                raise ValueError("previous_grid must match grid")
            object.__setattr__(self, "previous_grid", previous.copy())


@dataclass
class EntitySlot:
    entity_id: str
    colors: tuple[int, ...]
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: int
    shape: str
    relations: dict[str, tuple[str, ...]] = field(default_factory=dict)
    trajectory: list[tuple[int, float, float]] = field(default_factory=list)
    role_probabilities: dict[str, float] = field(default_factory=dict)
    symmetry: tuple[str, ...] = ()
    is_ui: bool = False


@dataclass
class RuleHypothesis:
    hypothesis_id: str
    program: str
    goal: str
    parameters: dict[str, Any] = field(default_factory=dict)
    supporting: list[str] = field(default_factory=list)
    contradicting: list[str] = field(default_factory=list)
    confidence: float = 0.25
    uncertainty: float = 0.75
    predictions: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class BeliefState:
    observation: Observation
    entities: list[EntitySlot]
    hypotheses: list[RuleHypothesis]
    state_id: str
    transition_graph: Any
    goals: dict[str, float]
    resources: dict[str, float]
    uncertainty: float
    legal_actions: tuple[str, ...]
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchBudget:
    simulations: int = 64
    milliseconds: int = 500
    hypothesis_width: int = 8
    beam_width: int = 16


@dataclass
class ActionPlan:
    actions: list[dict[str, Any]]
    expected_effects: list[dict[str, Any]]
    interruption_conditions: tuple[str, ...]
    confidence: float
    goal: str
    source: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionOutcome:
    action: dict[str, Any]
    before: Observation
    after: Observation
    terminal: bool = False
    reset: bool = False
    reward: float = 0.0


@dataclass
class BeliefUpdate:
    prediction_error: float
    meaningful_divergence: bool
    changed_pixels: int
    hypothesis_confidences: dict[str, float]


@runtime_checkable
class DecisionEngine(Protocol):
    def observe(self, observation: Observation) -> BeliefState: ...

    def plan(self, belief: BeliefState, budget: SearchBudget) -> ActionPlan: ...

    def acknowledge(self, outcome: ActionOutcome) -> BeliefUpdate: ...
