"""Immutable transition-shard generation for local or distributed training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .gym import GymConfig, MECHANIC_FAMILIES, ProceduralArcEnv, family_split


ACTION_INDEX = {f"ACTION{index}": index for index in range(1, 8)}


def generate_shard(path: Path, *, episodes: int, seed: int, split: str = "train") -> Path:
    grids, next_grids, actions, future_actions, rewards, dones, families = [], [], [], [], [], [], []
    selected = [family for family in MECHANIC_FAMILIES if family_split(family) == split]
    if not selected:
        raise ValueError(f"no families in split {split}")
    for episode in range(episodes):
        family = selected[episode % len(selected)]
        env = ProceduralArcEnv(GymConfig(family=family, seed=seed + episode))
        grid = env.grid.copy()
        episode_actions: list[int] = []
        while not env.done:
            action = env.oracle_action() if episode % 3 else _random_action(env)
            next_grid, reward, done, _ = env.step(action)
            grids.append(grid); next_grids.append(next_grid)
            action_index = ACTION_INDEX[action["action"]]
            actions.append(action_index); episode_actions.append(action_index)
            rewards.append(reward); dones.append(done); families.append(family)
            grid = next_grid
        for index in range(len(episode_actions)):
            future = episode_actions[index:index + 8]
            future_actions.append(future + [0] * (8 - len(future)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, grid=np.asarray(grids, dtype=np.uint8), next_grid=np.asarray(next_grids, dtype=np.uint8), action=np.asarray(actions, dtype=np.int8), future_action=np.asarray(future_actions, dtype=np.int8), reward=np.asarray(rewards, dtype=np.float32), done=np.asarray(dones, dtype=bool), family=np.asarray(families))
    return path


def _random_action(env: ProceduralArcEnv) -> dict[str, object]:
    name = str(env.rng.choice(env.available_actions))
    if name == "ACTION6":
        return {"action": name, "x": int(env.rng.integers(64)), "y": int(env.rng.integers(64))}
    return {"action": name, "x": None, "y": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["train", "heldout"], default="train")
    args = parser.parse_args()
    print(generate_shard(args.output, episodes=args.episodes, seed=args.seed, split=args.split))


if __name__ == "__main__":
    main()
