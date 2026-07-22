"""Parallel, compact, executable mechanic and goal hypotheses."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from .schemas import ActionOutcome, EntitySlot, RuleHypothesis


DEFAULT_HYPOTHESES = (
    ("navigation", "move_until_collision", "reach_salient_target"),
    ("collection", "contact_removes_object", "collect_before_goal"),
    ("matching", "align_matching_geometry", "match_shape_or_orientation"),
    ("transformation", "contact_transforms_geometry", "make_counterparts_equal"),
    ("manipulation", "click_changes_contained_object", "activate_affordance"),
    ("sequence", "repeat_effectful_actions", "execute_discovered_sequence"),
    ("resource", "edge_indicator_tracks_budget", "preserve_required_resource"),
    ("avoidance", "contact_causes_reset", "reach_goal_without_hazard"),
)


class HypothesisBank:
    def __init__(self, width: int = 8) -> None:
        self.width = max(1, min(8, width))
        self.items = [
            RuleHypothesis(name, program, goal, confidence=0.22, uncertainty=0.78)
            for name, program, goal in DEFAULT_HYPOTHESES[: self.width]
        ]

    def update(self, outcome: ActionOutcome, entities: Iterable[EntitySlot]) -> list[RuleHypothesis]:
        changed = int(np.count_nonzero(outcome.before.grid != outcome.after.grid))
        edge = self._edge_changed(outcome.before.grid, outcome.after.grid)
        for index, hypothesis in enumerate(self.items):
            support = False
            contradiction = False
            evidence = f"a{outcome.after.action_number}:{changed}px"
            if hypothesis.hypothesis_id == "navigation":
                support = 0 < changed < 512
                contradiction = changed == 0
            elif hypothesis.hypothesis_id in {"collection", "transformation", "manipulation", "sequence"}:
                support = changed > 0
            elif hypothesis.hypothesis_id == "resource":
                support = edge
            elif hypothesis.hypothesis_id == "avoidance":
                support = outcome.reset
                contradiction = changed > 0 and not outcome.reset
            elif hypothesis.hypothesis_id == "matching":
                support = any(entity.symmetry for entity in entities)
            confidence = hypothesis.confidence
            confidence = min(0.98, confidence + 0.12) if support else confidence
            confidence = max(0.02, confidence - 0.10) if contradiction else confidence
            supporting = hypothesis.supporting + ([evidence] if support else [])
            contradicting = hypothesis.contradicting + ([evidence] if contradiction else [])
            self.items[index] = replace(
                hypothesis,
                supporting=supporting[-12:],
                contradicting=contradicting[-12:],
                confidence=confidence,
                uncertainty=1.0 - confidence,
            )
        self.items.sort(key=lambda item: (-item.confidence, len(item.program), item.hypothesis_id))
        return list(self.items)

    @staticmethod
    def _edge_changed(before: np.ndarray, after: np.ndarray) -> bool:
        changed = before != after
        return bool(changed[0].any() or changed[-1].any() or changed[:, 0].any() or changed[:, -1].any())

    def disagreement(self) -> float:
        if not self.items:
            return 1.0
        values = np.asarray([item.confidence for item in self.items], dtype=np.float32)
        return float(min(1.0, values.std() * 4.0 + np.mean(1.0 - values)))
