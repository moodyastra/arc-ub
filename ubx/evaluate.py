"""Ablation-oriented procedural evaluation and generalization gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .engine import GraphBaselineEngine, OfflineUBXEngine
from .gym import GymConfig, MECHANIC_FAMILIES, ProceduralArcEnv, family_split
from .schemas import ActionOutcome, Observation, SearchBudget


def evaluate(engine_name: str, *, episodes_per_family: int, model_path: Path | None = None, disabled_experts: tuple[str, ...] = ()) -> dict[str, object]:
    by_family: dict[str, dict[str, float]] = {}
    all_times, repeats, dead_repeats = [], 0, 0
    for family in MECHANIC_FAMILIES:
        completed, actions_used = 0, []
        for episode in range(episodes_per_family):
            env = ProceduralArcEnv(GymConfig(family=family, seed=10_000 + episode))
            engine = GraphBaselineEngine() if engine_name == "graph_baseline" else OfflineUBXEngine(model_path, disabled_experts=disabled_experts)
            grid, previous = env.grid.copy(), None
            seen: set[bytes] = set()
            for step in range(env.config.max_steps):
                observation = Observation(grid, previous, env.available_actions, 1, step)
                belief = engine.observe(observation)
                started = time.perf_counter()
                plan = engine.plan(belief, SearchBudget(milliseconds=500))
                all_times.append((time.perf_counter() - started) * 1000)
                action = plan.actions[0]
                before = observation
                next_grid, reward, done, info = env.step(action)
                after = Observation(next_grid, grid, env.available_actions, 1, step + 1)
                update = engine.acknowledge(ActionOutcome(action, before, after, terminal=done and reward > 0, reset=bool(info["reset"]), reward=reward))
                signature = next_grid.tobytes()
                if signature in seen: repeats += 1
                if update.changed_pixels == 0 and signature in seen: dead_repeats += 1
                seen.add(signature)
                previous, grid = grid, next_grid
                if done:
                    if reward > 0: completed += 1
                    actions_used.append(step + 1)
                    break
        by_family[family] = {"completion": completed / episodes_per_family, "mean_actions": float(np.mean(actions_used)) if actions_used else float(env.config.max_steps), "split": family_split(family)}
    heldout = [value["completion"] for value in by_family.values() if value["split"] == "heldout"]
    return {
        "engine": engine_name,
        "families": by_family,
        "heldout_family_success_fraction": float(np.mean(np.asarray(heldout) > 0)) if heldout else 0.0,
        "mean_planning_ms": float(np.mean(all_times)) if all_times else 0.0,
        "repeated_states": repeats,
        "repeated_dead_actions": dead_repeats,
        "generalization_gate_80_percent": bool(heldout and np.mean(np.asarray(heldout) > 0) >= 0.8),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["graph_baseline", "offline_ubx"], default="graph_baseline")
    parser.add_argument("--episodes-per-family", type=int, default=5)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--disable-neural-expert", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.engine, episodes_per_family=args.episodes_per_family, model_path=args.model_path, disabled_experts=tuple(args.disable_neural_expert))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
