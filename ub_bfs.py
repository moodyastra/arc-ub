"""Bounded last-resort BFS over cloned ARC game state."""

from __future__ import annotations

import copy
import hashlib
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from arcengine import ActionInput, GameAction, GameState


@dataclass(frozen=True)
class BFSResult:
    found: bool
    actions: list[str]
    expanded: int
    generated: int
    elapsed_seconds: float
    stop_reason: str


def _hidden_state(game: object) -> tuple[object, ...]:
    """Capture small primitive and moving-helper state not guaranteed visible in a frame."""
    result: list[object] = []
    for name, value in sorted(vars(game).items()):
        if isinstance(value, (bool, int, float, str, type(None))):
            result.append((name, value))
            continue
        if isinstance(value, (list, tuple)) and len(value) <= 64:
            helpers: list[object] = []
            for item in value:
                sprite = getattr(item, "_sprite", None)
                direction = getattr(item, "_dir", None)
                if sprite is not None:
                    helpers.append(
                        (
                            getattr(sprite, "x", None),
                            getattr(sprite, "y", None),
                            getattr(sprite, "color", None),
                            getattr(sprite, "rotation", None),
                            direction,
                        )
                    )
                elif isinstance(item, (bool, int, float, str, type(None))):
                    helpers.append(item)
            if helpers:
                result.append((name, tuple(helpers)))
    return tuple(result)


def _state_key(
    game: object,
    observation: Any,
    ignore_regions_yxyx: Iterable[Iterable[int]],
) -> tuple[object, ...]:
    frame = observation.frame[-1].copy()
    for y0, y1, x0, x1 in ignore_regions_yxyx:
        frame[int(y0) : int(y1), int(x0) : int(x1)] = 0
    digest = hashlib.blake2b(frame.tobytes(), digest_size=16).digest()
    return (
        observation.levels_completed,
        observation.state.name,
        tuple(observation.available_actions),
        digest,
        _hidden_state(game),
    )


def find_level_completion(
    live_game: object,
    start_observation: Any,
    *,
    preferred_actions: Iterable[str] = (),
    ignore_regions_yxyx: Iterable[Iterable[int]] = (),
    max_depth: int = 20,
    max_generated: int = 500,
    max_frontier: int = 128,
    timeout_seconds: float = 5.0,
) -> BFSResult:
    """Find a shortest level-completing path without stepping the live wrapper or game."""
    started = time.perf_counter()
    root_game = copy.deepcopy(live_game)
    root_level = start_observation.levels_completed
    queue: deque[tuple[str, ...]] = deque([tuple()])
    seen: set[tuple[object, ...]] = set()
    expanded = 0
    generated = 1

    preferred_ids = []
    for name in preferred_actions:
        if name.startswith("ACTION") and name[6:].isdigit():
            action_id = int(name[6:])
            if action_id not in preferred_ids:
                preferred_ids.append(action_id)

    while queue:
        elapsed = time.perf_counter() - started
        if elapsed >= timeout_seconds:
            return BFSResult(False, [], expanded, generated, elapsed, "timeout")
        if generated >= max_generated:
            return BFSResult(False, [], expanded, generated, elapsed, "generated_cap")

        path = queue.popleft()
        game = copy.deepcopy(root_game)
        observation = start_observation
        valid = True
        for action_name in path:
            observation = game.perform_action(
                ActionInput(id=GameAction.from_id(int(action_name[6:])), data={}),
                raw=True,
            )
            if observation is None or observation.state == GameState.GAME_OVER:
                valid = False
                break
        if not valid:
            continue

        expanded += 1
        if observation.state == GameState.WIN or observation.levels_completed > root_level:
            return BFSResult(
                True,
                list(path),
                expanded,
                generated,
                time.perf_counter() - started,
                "level_completed",
            )

        key = _state_key(game, observation, ignore_regions_yxyx)
        if key in seen:
            continue
        seen.add(key)
        if len(path) >= max_depth:
            continue

        available_ids = [int(value) for value in observation.available_actions]
        action_ids = [value for value in preferred_ids if value in available_ids]
        action_ids.extend(value for value in available_ids if value not in action_ids)
        action_ids = [value for value in action_ids if value != 6]
        for action_id in action_ids:
            if len(queue) >= max_frontier:
                return BFSResult(
                    False,
                    [],
                    expanded,
                    generated,
                    time.perf_counter() - started,
                    "frontier_cap",
                )
            queue.append((*path, f"ACTION{action_id}"))
            generated += 1

    return BFSResult(
        False,
        [],
        expanded,
        generated,
        time.perf_counter() - started,
        "exhausted",
    )
