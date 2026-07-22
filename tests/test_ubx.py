from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ubx.engine import GraphBaselineEngine
from ubx.graph import TransitionGraph
from ubx.gym import GymConfig, MECHANIC_FAMILIES, ProceduralArcEnv, family_split
from ubx.perception import MultiviewPerception
from ubx.schemas import ActionOutcome, Observation, SearchBudget


class SchemaTests(unittest.TestCase):
    def test_grid_contract_is_exact(self) -> None:
        with self.assertRaises(ValueError):
            Observation(np.zeros((8, 8)), None, ("ACTION1",))
        with self.assertRaises(ValueError):
            Observation(np.full((64, 64), 16), None, ("ACTION1",))


class PerceptionTests(unittest.TestCase):
    def test_all_views_and_ui_are_preserved(self) -> None:
        grid = np.zeros((64, 64), dtype=np.uint8)
        grid[1, 2:10] = 11
        grid[30:33, 30:33] = 9
        entities, views = MultiviewPerception().encode(Observation(grid, None, ("ACTION1",)))
        self.assertEqual(views["patch_2x2"].shape, (32, 32, 16))
        self.assertEqual(views["patch_4x4"].shape, (16, 16, 16))
        self.assertTrue(any(entity.is_ui and 11 in entity.colors for entity in entities))


class GraphTests(unittest.TestCase):
    def test_dead_action_and_frontier(self) -> None:
        graph = TransitionGraph()
        grid = np.zeros((64, 64), dtype=np.uint8)
        state = graph.remember(grid)
        graph.add(grid, "ACTION1", grid, terminal=False, reset=False)
        self.assertEqual(graph.dead_actions(state), {"ACTION1"})
        self.assertEqual(graph.frontier_actions(state, ("ACTION1", "ACTION2")), ["ACTION2"])


class EngineTests(unittest.TestCase):
    def test_first_three_are_checked_then_macro_is_allowed(self) -> None:
        engine = GraphBaselineEngine()
        grid = np.zeros((64, 64), dtype=np.uint8)
        for step in range(3):
            observation = Observation(grid, None, ("ACTION1", "ACTION2", "ACTION3", "ACTION4"), action_number=step)
            plan = engine.plan(engine.observe(observation), SearchBudget(milliseconds=30))
            self.assertEqual(len(plan.actions), 1)
        observation = Observation(grid, None, ("ACTION1", "ACTION2", "ACTION3", "ACTION4"), action_number=3)
        belief = engine.observe(observation)
        belief.uncertainty = 0.4
        plan = engine.plan(belief, SearchBudget(milliseconds=30))
        self.assertGreaterEqual(len(plan.actions), 3)

    def test_prediction_divergence_is_reported(self) -> None:
        engine = GraphBaselineEngine()
        before_grid = np.zeros((64, 64), dtype=np.uint8)
        after_grid = before_grid.copy(); after_grid[3, 3] = 2
        before = Observation(before_grid, None, ("ACTION1",))
        belief = engine.observe(before)
        engine.plan(belief, SearchBudget(milliseconds=20))
        after = Observation(after_grid, before_grid, ("ACTION1",), action_number=1)
        update = engine.acknowledge(ActionOutcome({"action": "ACTION1", "x": None, "y": None}, before, after))
        self.assertEqual(update.changed_pixels, 1)


class GymTests(unittest.TestCase):
    def test_families_have_train_and_holdout_split(self) -> None:
        splits = {family_split(family) for family in MECHANIC_FAMILIES}
        self.assertEqual(splits, {"train", "heldout"})

    def test_click_family_is_solvable(self) -> None:
        env = ProceduralArcEnv(GymConfig(family="click", seed=4))
        _, reward, done, _ = env.step(env.oracle_action())
        self.assertTrue(done)
        self.assertGreater(reward, 0)


if __name__ == "__main__":
    unittest.main()
