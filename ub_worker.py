"""Execute validated Luna Light commands in ARC and return fresh screenshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import arc_agi
import numpy as np
from arc_agi.rendering import frame_to_rgb_array
from arcengine import GameAction, GameState
from PIL import Image

from ub import GameCommand, LunaDecision, LunaLightCLIPlanner
from ub_bfs import BFSResult, find_level_completion
from ub_object_map import ObjectMapStore
from ub_scene import SceneAnalyzer, snapshot_to_markdown
from ubx.adapter import OfflinePlannerAdapter


WORKSPACE = Path(__file__).resolve().parent


def load_arc_api_key(workspace: Path = WORKSPACE) -> tuple[str, str]:
    """Load the user's persistent ARC profile key without creating or printing one."""
    for path in (workspace / ".env2", workspace / ".env2.txt"):
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                name, value = line.split("=", 1)
                if name.strip().removeprefix("export ").strip() != "ARC_API_KEY":
                    continue
                key = value.strip().strip('"\'')
            else:
                key = line.strip('"\'')
            if key:
                os.environ["ARC_API_KEY"] = key
                return key, path.name
    inherited = os.environ.get("ARC_API_KEY", "").strip()
    if inherited:
        return inherited, os.environ.get("ARC_API_KEY_SOURCE", "environment")
    raise RuntimeError(
        "ARC_API_KEY is missing. Put the existing key in .env2 or .env2.txt "
        "as either a raw line or ARC_API_KEY=<key>."
    )


def frame_difference(before: np.ndarray, after: np.ndarray) -> dict[str, object]:
    """Summarize whole-frame and side-UI changes without interpreting the scene."""
    changed = before != after
    points = np.argwhere(changed)
    bbox: list[int] | None = None
    if len(points):
        y0, x0 = points.min(axis=0)
        y1, x1 = points.max(axis=0)
        bbox = [int(x0), int(y0), int(x1), int(y1)]

    height, width = before.shape
    edge_width = 8
    edge_mask = np.zeros_like(changed, dtype=bool)
    edge_mask[:edge_width, :] = True
    edge_mask[-edge_width:, :] = True
    edge_mask[:, :edge_width] = True
    edge_mask[:, -edge_width:] = True

    edge_color_delta: dict[str, int] = {}
    for color in range(16):
        delta = int(np.count_nonzero(after[edge_mask] == color) - np.count_nonzero(before[edge_mask] == color))
        if delta:
            edge_color_delta[str(color)] = delta

    return {
        "changed_pixels": int(np.count_nonzero(changed)),
        "change_bbox_xyxy": bbox,
        "edge_changed_pixels": int(np.count_nonzero(changed & edge_mask)),
        "edge_color_count_delta": edge_color_delta,
    }


def playfield_changed_pixels(
    before: np.ndarray,
    after: np.ndarray,
    ignored_regions_yxyx: list[list[int]] | None = None,
) -> int:
    """Count changed cells after masking known volatile HUD regions."""
    changed = before != after
    height, width = changed.shape
    for y0, y1, x0, x1 in ignored_regions_yxyx or []:
        changed[
            max(0, int(y0)) : min(height, int(y1)),
            max(0, int(x0)) : min(width, int(x1)),
        ] = False
    return int(np.count_nonzero(changed))


class UBWorker:
    def __init__(
        self,
        game_id: str,
        planner: LunaLightCLIPlanner | OfflinePlannerAdapter,
        *,
        escalation_planner: LunaLightCLIPlanner | None = None,
        run_root: Path,
        memory_root: Path,
        max_actions: int,
        max_planner_turns: int,
        max_batch: int,
        target_levels: int | None,
        max_restarts: int,
        fresh_memory: bool,
        arc_api_key: str,
        arc_key_source: str,
        step_delay: float = 0.0,
        competition: bool = False,
    ) -> None:
        self.game_id = game_id
        self.planner = planner
        self.escalation_planner = escalation_planner
        self.max_actions = max_actions
        self.max_planner_turns = max_planner_turns
        self.max_batch = max(3, min(max_batch, 8))
        self.target_levels = target_levels
        self.max_restarts = max(0, max_restarts)
        self.step_delay = max(0.0, step_delay)
        self.total_actions = 0
        self.actions_in_level = 0
        self.planner_turns = 0
        self.restarts = 0
        self.bfs_attempts = 0
        self.sol_escalations = 0
        self.sol_consultations = 0
        self.sol_escalations_in_level = 0
        self.consecutive_stuck_turns = 0
        self.last_escalation_planner_turn = -1_000_000
        self.last_feedback: list[dict[str, object]] = []
        self.action_history: list[str] = []
        self.last_planner_decision: dict[str, object] | None = None
        self.recent_decisions: list[dict[str, object]] = []
        self.bfs_state_attempts: set[str] = set()
        self.escalation_state_attempts: set[str] = set()
        self.semantic_state_visits: dict[str, int] = {}
        self.rules = planner.rules
        self.bfs_policy = self.rules["execution_policy"]["bfs"]
        self.escalation_policy = self.rules["execution_policy"].get("escalation", {})
        self.game_profile = self.rules.get("game_profiles", {}).get(game_id, {})
        self.visual_feed_policy = self.rules["execution_policy"].get("visual_feed", {})
        self.arc_key_source = arc_key_source

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = run_root / f"{game_id}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.trace_path = self.run_dir / "trace.jsonl"
        self.object_map = ObjectMapStore(
            game_id,
            self.run_dir,
            memory_root,
            fresh=fresh_memory,
        )

        operation_mode = arc_agi.OperationMode.COMPETITION if competition else arc_agi.OperationMode.NORMAL
        self.arcade = arc_agi.Arcade(arc_api_key=arc_api_key, operation_mode=operation_mode)
        operation_mode = getattr(self.arcade, "operation_mode", None)
        self.arc_operation_mode = str(
            getattr(operation_mode, "value", operation_mode or "unknown")
        )
        configured = getattr(self.arcade, "arc_api_key", None) == arc_api_key
        self.arc_profile_status = (
            "configured_unverified" if configured else "key_mismatch"
        )
        self.environment = self.arcade.make(game_id)
        observation = self.environment.step(GameAction.RESET)
        if observation is None:
            raise RuntimeError("ARC environment returned no reset observation")
        self.observation = observation
        self.current_frame = observation.frame[-1].copy()
        self.level_overview_shot: Path | None = None
        self.last_checked_shot: Path | None = None
        self.scene_analyzer = SceneAnalyzer()
        self.scene_json_path = self.run_dir / "scene_inventory.json"
        self.scene_markdown_path = self.run_dir / "scene_inventory.md"
        self.scene_events_path = self.run_dir / "scene_events.jsonl"
        self.scene_snapshot = self._update_scene(0)

    def _record(self, event: dict[str, object]) -> None:
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _capture(self, suffix: str) -> Path:
        path = self.run_dir / f"step_{self.total_actions:03d}_{suffix}.png"
        rgb = frame_to_rgb_array(self.total_actions, self.current_frame, scale=10)
        Image.fromarray(rgb).save(path)
        return path

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)

    def _update_scene(self, action_number: int) -> dict[str, object]:
        """Persist a complete census plus a small event-driven delta packet."""
        snapshot = self.scene_analyzer.analyze(self.current_frame, action_number)
        snapshot["level_number"] = self._current_level_number()
        self._apply_profile_ui_roles(snapshot)
        # The generic analyzer cannot know game-specific HUD bounds. Rebuild only
        # the compact semantic layer after those labels are applied; the raw census
        # itself remains unchanged for diagnostics and replay.
        snapshot["perceptual_summary"] = self.scene_analyzer.summarize(snapshot)
        self._atomic_write(
            self.scene_json_path,
            json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        )
        self._atomic_write(self.scene_markdown_path, snapshot_to_markdown(snapshot))

        changed_ids = {
            stable_id
            for event, stable_ids in snapshot["changes"].items()
            if event not in {"unchanged", "disappeared"}
            for stable_id in stable_ids
        }
        delta = {
            "level_number": snapshot["level_number"],
            "action_number": snapshot["action_number"],
            "component_count": snapshot["component_count"],
            "events": {
                event: values
                for event, values in snapshot["changes"].items()
                if event != "unchanged" and values
            },
            "color_deltas": snapshot["color_deltas"],
            "changed_components": [
                component
                for component in snapshot["components"]
                if component["id"] in changed_ids
            ],
        }
        with self.scene_events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(delta, ensure_ascii=False) + "\n")
        self.scene_delta = delta
        self.scene_snapshot = snapshot
        return snapshot

    def _apply_profile_ui_roles(self, snapshot: dict[str, object]) -> None:
        declared = self.game_profile.get(
            "ui_regions_yxyx",
            self.game_profile.get("bfs_ignore_regions_yxyx", []),
        )
        for item in snapshot.get("components", []):
            if not isinstance(item, dict):
                continue
            item["role_source"] = "geometry_hint"
            if item.get("role_hint") == "terrain/background":
                continue
            x0, y0, x1, y1 = (int(value) for value in item["bbox"])
            area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
            for ui_y0, ui_y1, ui_x0, ui_x1 in declared:
                overlap_width = max(0, min(x1 + 1, int(ui_x1)) - max(x0, int(ui_x0)))
                overlap_height = max(0, min(y1 + 1, int(ui_y1)) - max(y0, int(ui_y0)))
                if overlap_width * overlap_height / area >= 0.5:
                    item["role_hint"] = "edge_ui"
                    item["role_source"] = "game_profile"
                    break

    def _compact_scene_inventory(self) -> dict[str, object]:
        """Complete but token-conscious current scene supplied to each planner."""
        components = []
        for item in self.scene_snapshot.get("components", []):
            components.append(
                {
                    "id": item["id"],
                    "color": item["color_name"],
                    "bbox_xyxy": item["bbox"],
                    "size_wh": [item["width"], item["height"]],
                    "area": item["area"],
                    "shape_kind": item["shape_kind"],
                    "shape": item["shape_signature"],
                    "orientation": item["orientation"],
                    "fill_ratio": item["fill_ratio"],
                    "symmetry": item["symmetry"],
                    "rotation_class": item["rotation_signature"],
                    "holes": item["holes"],
                    "edges": item["edge_contacts"],
                    "role_hint": item["role_hint"],
                    "role_source": item.get("role_source", "geometry_hint"),
                    "relations": item["relations"],
                    "change": item["change"],
                }
            )
        return {
            "level_number": self.scene_snapshot.get("level_number"),
            "action_number": self.scene_snapshot.get("action_number"),
            "frame": self.scene_snapshot.get("frame"),
            "component_count": self.scene_snapshot.get("component_count"),
            "colors_present": self.scene_snapshot.get("colors_present"),
            "color_deltas": self.scene_snapshot.get("color_deltas"),
            "components": components,
            "disappeared": self.scene_snapshot.get("changes", {}).get("disappeared", []),
        }

    def _perceptual_scene_summary(self) -> dict[str, object]:
        """Return grouped human-facing evidence, never the raw component census."""
        summary = dict(self.scene_snapshot.get("perceptual_summary", {}))
        summary.update(
            {
                "level_number": self.scene_snapshot.get("level_number"),
                "action_number": self.scene_snapshot.get("action_number"),
                "frame": self.scene_snapshot.get("frame"),
                "colors_present": self.scene_snapshot.get("colors_present"),
            }
        )
        return summary

    def _perceptual_scene_delta(self) -> dict[str, object]:
        """Collapse raw component changes into changed semantic groups and colors."""
        changed_ids: set[str] = set()
        disappeared: list[dict[str, object]] = []
        for event, values in self.scene_delta.get("events", {}).items():
            for value in values:
                if isinstance(value, dict):
                    stable_id = str(value.get("id", ""))
                    if event == "disappeared":
                        disappeared.append(value)
                else:
                    stable_id = str(value)
                if stable_id:
                    changed_ids.add(stable_id)

        summary = self.scene_snapshot.get("perceptual_summary", {})
        changed_candidates = [
            candidate
            for candidate in summary.get("candidates", [])
            if changed_ids & {str(value) for value in candidate.get("component_ids", [])}
        ]
        background_ids = {
            str(value)
            for value in summary.get("background", {}).get("component_ids", [])
        }
        return {
            "level_number": self.scene_delta.get("level_number"),
            "action_number": self.scene_delta.get("action_number"),
            "color_deltas": self.scene_delta.get("color_deltas", []),
            "changed_candidates": changed_candidates,
            "disappeared_components": disappeared,
            "background_changed": bool(changed_ids & background_ids),
        }

    @staticmethod
    def _edge_ui_regions(snapshot: dict[str, object]) -> list[list[int]]:
        regions = []
        for item in snapshot.get("components", []):
            if not isinstance(item, dict) or item.get("role_hint") != "edge_ui":
                continue
            x0, y0, x1, y1 = item["bbox"]
            regions.append([int(y0), int(y1) + 1, int(x0), int(x1) + 1])
        return regions

    def _ignored_ui_regions(
        self, *snapshots: dict[str, object]
    ) -> list[list[int]]:
        regions = [
            list(region)
            for region in self.game_profile.get(
                "ui_regions_yxyx",
                self.game_profile.get("bfs_ignore_regions_yxyx", []),
            )
        ]
        for snapshot in snapshots:
            regions.extend(self._edge_ui_regions(snapshot))
        return regions

    @staticmethod
    def _player_component_ids(
        decision: LunaDecision, snapshot: dict[str, object]
    ) -> set[str]:
        player_words = {"player", "pawn", "avatar", "character"}
        observations = [
            item
            for item in decision.objects
            if set(re.findall(r"[a-z0-9]+", item.real_life_possible_object.lower()))
            & player_words
        ]
        if not observations:
            return set()
        player = max(observations, key=lambda item: item.importance)
        px0, px1 = player.position.x_min, player.position.x_max
        py0, py1 = player.position.y_min, player.position.y_max
        color_words = set(re.findall(r"[a-z]+", player.color.lower()))
        candidates: list[tuple[bool, float, str]] = []
        for component in snapshot.get("components", []):
            if not isinstance(component, dict):
                continue
            if component.get("role_hint") in {
                "terrain/background",
                "edge_ui",
                "edge_region_candidate",
            }:
                continue
            x0, y0, x1, y1 = (int(value) for value in component["bbox"])
            overlap_width = max(0, min(px1, x1) - max(px0, x0) + 1)
            overlap_height = max(0, min(py1, y1) - max(py0, y0) + 1)
            overlap = overlap_width * overlap_height
            if overlap == 0:
                continue
            component_area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
            component_color_words = set(
                re.findall(r"[a-z]+", str(component.get("color_name", "")).lower())
            )
            color_match = bool(component_color_words & color_words)
            candidates.append((color_match, overlap / component_area, str(component["id"])))
        if not candidates:
            return set()
        if any(color_match for color_match, _, _ in candidates):
            return {
                stable_id
                for color_match, overlap, stable_id in candidates
                if color_match and overlap >= 0.25
            }
        best_overlap = max(overlap for _, overlap, _ in candidates)
        return {
            stable_id
            for _, overlap, stable_id in candidates
            if overlap == best_overlap and overlap >= 0.5
        }

    @staticmethod
    def _player_motion(
        player_ids: set[str], snapshot: dict[str, object]
    ) -> bool | None:
        if not player_ids:
            return None
        current = {
            str(item["id"]): item
            for item in snapshot.get("components", [])
            if isinstance(item, dict)
        }
        present = [current[stable_id] for stable_id in player_ids if stable_id in current]
        if not present:
            return None
        return any("moved" in item["change"]["events"] for item in present)

    @staticmethod
    def _important_scene_events(
        before: dict[str, object],
        after: dict[str, object],
        player_ids: set[str],
    ) -> list[str]:
        important_changes = {"new", "rotated", "color_changed"}
        events: list[str] = []
        before_by_id = {
            str(item["id"]): item
            for item in before.get("components", [])
            if isinstance(item, dict)
        }
        for item in after.get("components", []):
            if not isinstance(item, dict) or str(item["id"]) in player_ids:
                continue
            if item.get("role_hint") != "object_candidate":
                continue
            changes = important_changes & set(item["change"]["events"])
            previous = before_by_id.get(str(item["id"]))
            if previous is not None and set(item["change"]["events"]) & {
                "geometry_changed",
                "resized",
            }:
                if (
                    previous.get("shape_kind") != item.get("shape_kind")
                    or previous.get("orientation") != item.get("orientation")
                    or previous.get("holes") != item.get("holes")
                ):
                    changes.add("structural_change")
            if changes:
                events.append(f"{item['id']}:{'+'.join(sorted(changes))}")

        for disappeared in after.get("changes", {}).get("disappeared", []):
            stable_id = str(disappeared["id"])
            previous = before_by_id.get(stable_id)
            if (
                stable_id not in player_ids
                and previous is not None
                and previous.get("role_hint") == "object_candidate"
            ):
                events.append(f"{stable_id}:disappeared")
        return events

    def _finished(self) -> bool:
        if self.observation.state == GameState.WIN:
            return True
        if self.target_levels is not None:
            return self.observation.levels_completed >= self.target_levels
        return self.observation.levels_completed >= self.observation.win_levels

    def _current_level_number(self) -> int:
        return int(self.observation.levels_completed) + 1

    def _bfs_mature(self, decision: LunaDecision) -> bool:
        thresholds = self.bfs_policy["maturity_any_of"]
        maturity_signal = (
            self.actions_in_level >= int(thresholds["actions_in_level"])
            or decision.scene_coverage >= float(thresholds["scene_coverage"])
            or decision.level_model_confidence >= float(thresholds["level_model_confidence"])
        )
        return (
            self.actions_in_level >= int(self.bfs_policy["minimum_checked_actions"])
            and maturity_signal
        )

    def _bfs_state_id(self) -> str:
        return self._semantic_state_id()

    def _semantic_state_id(self) -> str:
        """Hash playfield and HUD separately so resource changes remain meaningful."""
        frame = self.current_frame.copy()
        height, width = frame.shape
        ui_mask = np.zeros_like(frame, dtype=bool)
        normalized_regions = set()
        for y0, y1, x0, x1 in self._ignored_ui_regions(self.scene_snapshot):
            region = (
                max(0, int(y0)),
                min(height, int(y1)),
                max(0, int(x0)),
                min(width, int(x1)),
            )
            normalized_regions.add(region)
            ry0, ry1, rx0, rx1 = region
            ui_mask[ry0:ry1, rx0:rx1] = True

        hud_regions = []
        for y0, y1, x0, x1 in sorted(normalized_regions):
            values, counts = np.unique(frame[y0:y1, x0:x1], return_counts=True)
            hud_regions.append(
                [y0, y1, x0, x1, [[int(value), int(count)] for value, count in zip(values, counts)]]
            )
        frame[ui_mask] = 0
        playfield_digest = hashlib.blake2b(frame.tobytes(), digest_size=12).hexdigest()
        hud_payload = json.dumps(
            {
                "regions": hud_regions,
                "environment_state": self.observation.state.name,
                "available_actions": list(self.observation.available_actions),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        hud_digest = hashlib.blake2b(hud_payload, digest_size=8).hexdigest()
        return f"L{self._current_level_number()}:{playfield_digest}:{hud_digest}"

    @staticmethod
    def _decision_summary(decision: LunaDecision) -> dict[str, object]:
        return {
            "differences": decision.differences,
            "similarities": decision.similarities,
            "revised_hypotheses": decision.revised_hypotheses,
            "commands": [command.model_dump(mode="json") for command in decision.commands],
            "goal_status": decision.goal_status,
            "confidence": decision.confidence,
            "scene_coverage": decision.scene_coverage,
            "level_model_confidence": decision.level_model_confidence,
            "stuck": decision.stuck,
            "request_bfs": decision.request_bfs,
            "bfs_reason": decision.bfs_reason,
        }

    def _note_primary_decision(self, decision: LunaDecision) -> str:
        state_id = self._semantic_state_id()
        self.semantic_state_visits[state_id] = self.semantic_state_visits.get(state_id, 0) + 1
        self.consecutive_stuck_turns = (
            self.consecutive_stuck_turns + 1 if decision.stuck else 0
        )
        summary = self._decision_summary(decision)
        summary.update(
            {
                "planner_turn": self.planner_turns,
                "actions_in_level": self.actions_in_level,
                "semantic_state_id": state_id,
            }
        )
        self.recent_decisions.append(summary)
        keep = max(2, int(self.escalation_policy.get("history_decisions", 6)) * 2)
        self.recent_decisions = self.recent_decisions[-keep:]
        return state_id

    def _escalation_trigger(
        self,
        decision: LunaDecision,
        bfs_result: BFSResult | None,
        state_id: str,
    ) -> str | None:
        policy = self.escalation_policy
        if not policy.get("enabled", False) or self.escalation_planner is None:
            return None
        if self.observation.state in {GameState.GAME_OVER, GameState.WIN}:
            return None
        if bfs_result is not None and bfs_result.found:
            return None
        if state_id in self.escalation_state_attempts:
            return None
        if self.sol_escalations_in_level >= int(policy.get("max_attempts_per_level", 2)):
            return None

        if (
            bfs_result is not None
            and not bfs_result.found
            and policy.get("failed_bfs_allows_immediate", True)
        ):
            return f"bounded_bfs_failed:{bfs_result.stop_reason}"

        if self.actions_in_level < int(policy.get("minimum_actions_in_level", 10)):
            return None
        if not self._bfs_mature(decision):
            return None
        cooldown = int(policy.get("cooldown_planner_turns", 6))
        if self.planner_turns - self.last_escalation_planner_turn < cooldown:
            return None

        required_stuck = int(policy.get("required_consecutive_stuck_turns", 2))
        if self.consecutive_stuck_turns >= required_stuck:
            return f"stuck_reported_{self.consecutive_stuck_turns}_consecutive_checks"
        required_visits = int(policy.get("repeated_semantic_state_visits", 3))
        if self.semantic_state_visits.get(state_id, 0) >= required_visits:
            return f"semantic_state_revisited_{self.semantic_state_visits[state_id]}_times"
        return None

    def _build_escalation_evidence(
        self,
        *,
        trigger: str,
        decision: LunaDecision,
        bfs_result: BFSResult | None,
        state_id: str,
        allowed_commands: int,
        screenshots: list[tuple[str, Path]],
        prior_feedback: list[dict[str, object]],
    ) -> dict[str, object]:
        action_history_limit = int(self.escalation_policy.get("history_actions", 60))
        decision_history_limit = int(self.escalation_policy.get("history_decisions", 6))
        primary_summary = self._decision_summary(decision)
        primary_summary["objects"] = [
            observed.model_dump(mode="json") for observed in decision.objects
        ]
        return {
            "schema_version": 1,
            "purpose": "Compact last-resort gameplay recovery evidence.",
            "trigger": trigger,
            "semantic_state_id": state_id,
            "game": {
                "game_id": self.observation.game_id,
                "arc_profile_status": self.arc_profile_status,
                "arc_key_source": self.arc_key_source,
                "arc_operation_mode": self.arc_operation_mode,
                "scorecard_id": getattr(self.environment, "scorecard_id", None),
                "environment_state": self.observation.state.name,
                "current_level": self._current_level_number(),
                "levels_completed": self.observation.levels_completed,
                "win_levels": self.observation.win_levels,
            },
            "limits": {
                "available_actions": [
                    f"ACTION{value}" for value in self.observation.available_actions
                ],
                "maximum_commands_this_turn": allowed_commands,
                "control_hints": self.game_profile.get("controls", {}),
                "first_three_rule_active": self.actions_in_level < 3,
            },
            "progress": {
                "global_actions": self.total_actions,
                "actions_in_level": self.actions_in_level,
                "planner_turn": self.planner_turns,
                "semantic_state_visits": self.semantic_state_visits.get(state_id, 0),
                "consecutive_stuck_turns": self.consecutive_stuck_turns,
                "escalations_started_this_level": self.sol_escalations_in_level,
            },
            "recent_actions": self.action_history[-action_history_limit:],
            "recent_checked_feedback": prior_feedback[-5:],
            "perceptual_scene_summary": self._perceptual_scene_summary(),
            "perceptual_scene_delta": self._perceptual_scene_delta(),
            "current_level_object_map": self.object_map.compact_level(
                self._current_level_number(),
                max_rows=int(self.rules["object_memory"]["compact_max_rows"]),
            ),
            "primary_decision": primary_summary,
            "recent_decision_trend": self.recent_decisions[-decision_history_limit:],
            "bounded_bfs_result": asdict(bfs_result) if bfs_result is not None else None,
            "screenshots": [
                {"label": label, "file": path.name} for label, path in screenshots
            ],
        }

    def _run_sol_escalation(
        self,
        *,
        decision: LunaDecision,
        bfs_result: BFSResult | None,
        state_id: str,
        allowed_commands: int,
        screenshots: list[tuple[str, Path]],
        prior_feedback: list[dict[str, object]],
    ) -> tuple[LunaDecision, str, Path] | None:
        trigger = self._escalation_trigger(decision, bfs_result, state_id)
        if trigger is None or self.escalation_planner is None:
            return None

        self.sol_consultations += 1
        evidence = self._build_escalation_evidence(
            trigger=trigger,
            decision=decision,
            bfs_result=bfs_result,
            state_id=state_id,
            allowed_commands=allowed_commands,
            screenshots=screenshots,
            prior_feedback=prior_feedback,
        )
        evidence_path = self.run_dir / (
            f"sol_escalation_step_{self.total_actions:03d}_{self.sol_consultations:02d}.json"
        )
        temp_path = evidence_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        temp_path.replace(evidence_path)

        sol_context = {
            "planner_mode": "last_resort_recovery",
            "coordinate_system": self.rules["coordinate_system"],
            "available_actions": evidence["limits"]["available_actions"],
            "maximum_commands_this_turn": allowed_commands,
            "control_hints": self.game_profile.get("controls", {}),
            "escalation_evidence": evidence,
        }
        try:
            result = self.escalation_planner.decide_fresh(screenshots, sol_context)
        except (RuntimeError, ValueError) as exc:
            self.last_escalation_planner_turn = self.planner_turns
            event = {
                "type": "sol_escalation_failed",
                "trigger": trigger,
                "state_id": state_id,
                "model": self.escalation_planner.model,
                "reasoning_effort": self.escalation_planner.reasoning_effort,
                "evidence": evidence_path.name,
                "error": str(exc),
            }
            self._record(event)
            self.last_feedback.append(event)
            return None

        self.escalation_state_attempts.add(state_id)
        self.sol_escalations += 1
        self.sol_escalations_in_level += 1
        self.last_escalation_planner_turn = self.planner_turns
        event = {
            "type": "sol_escalation",
            "trigger": trigger,
            "state_id": state_id,
            "model": self.escalation_planner.model,
            "reasoning_effort": self.escalation_planner.reasoning_effort,
            "thread_id": result.thread_id,
            "evidence": evidence_path.name,
            "decision": result.decision.model_dump(mode="json"),
        }
        self._record(event)
        return result.decision, result.thread_id, evidence_path

    def _run_bfs(self, decision: LunaDecision) -> BFSResult | None:
        if not decision.request_bfs:
            return None

        state_id = self._bfs_state_id()
        mature = self._bfs_mature(decision)
        if not mature:
            event = {
                "type": "bfs_denied",
                "state_id": state_id,
                "actions_in_level": self.actions_in_level,
                "scene_coverage": decision.scene_coverage,
                "level_model_confidence": decision.level_model_confidence,
                "reason": "first-three or maturity gate not satisfied",
            }
            self._record(event)
            self.last_feedback.append(event)
            return None
        if state_id in self.bfs_state_attempts:
            event = {
                "type": "bfs_denied",
                "state_id": state_id,
                "reason": "this exact state was already searched",
            }
            self._record(event)
            self.last_feedback.append(event)
            return None

        self.bfs_state_attempts.add(state_id)
        self.bfs_attempts += 1
        preferred = [command.action for command in decision.commands]
        result = find_level_completion(
            self.environment._game,
            self.observation,
            preferred_actions=preferred,
            ignore_regions_yxyx=self._ignored_ui_regions(self.scene_snapshot),
            max_depth=int(self.bfs_policy["max_depth"]),
            max_generated=int(self.bfs_policy["max_generated_states"]),
            max_frontier=int(self.bfs_policy["max_frontier"]),
            timeout_seconds=float(self.bfs_policy["timeout_seconds"]),
        )
        event = {
            "type": "bfs_result",
            "state_id": state_id,
            "requested_because": decision.bfs_reason,
            "found": result.found,
            "path": result.actions,
            "expanded": result.expanded,
            "generated": result.generated,
            "elapsed_seconds": result.elapsed_seconds,
            "stop_reason": result.stop_reason,
        }
        self._record(event)
        self.last_feedback.append(event)
        return result

    def _context(
        self, allowed_commands: int, *, include_full_scene: bool = False
    ) -> dict[str, object]:
        available = [f"ACTION{value}" for value in self.observation.available_actions]
        context: dict[str, object] = {
            "game_id": self.observation.game_id,
            "environment_state": self.observation.state.name,
            "levels_completed": self.observation.levels_completed,
            "win_levels": self.observation.win_levels,
            "current_level": self._current_level_number(),
            "global_actions_taken": self.total_actions,
            "actions_taken_in_current_level": self.actions_in_level,
            "planner_turn": self.planner_turns + 1,
            "available_actions": available,
            "control_hints": self.game_profile.get("controls", {}),
            "maximum_commands_this_turn": allowed_commands,
            "recommended_batch_size_after_first_three": 3,
            "minimum_commands_this_turn": 1 if self.actions_in_level < 3 else 3,
            "first_three_rule_active": self.actions_in_level < 3,
            "short_batch_exception": self.rules["execution_policy"].get(
                "short_batch_exception", {}
            ),
            "bfs_gate": {
                "minimum_checked_actions": self.bfs_policy["minimum_checked_actions"],
                "actions_threshold": self.bfs_policy["maturity_any_of"]["actions_in_level"],
                "scene_coverage_threshold": self.bfs_policy["maturity_any_of"]["scene_coverage"],
                "understanding_threshold": self.bfs_policy["maturity_any_of"]["level_model_confidence"],
                "eligible_by_action_count_now": self.actions_in_level
                >= int(self.bfs_policy["maturity_any_of"]["actions_in_level"]),
            },
            "coordinate_system": self.rules["coordinate_system"],
            "current_level_object_map": self.object_map.compact_level(
                self._current_level_number(),
                max_rows=int(self.rules["object_memory"]["compact_max_rows"]),
            ),
            "recent_action_history": self.action_history[-30:],
            "last_worker_feedback": self.last_feedback,
            "last_planner_decision": self.last_planner_decision,
            "perceptual_scene_summary": self._perceptual_scene_summary(),
            "perceptual_scene_delta": self._perceptual_scene_delta(),
        }
        if getattr(self.planner, "needs_exact_grid", False):
            context["exact_grid"] = self.current_frame.astype(int).tolist()
        if include_full_scene:
            context["scene_observation_mode"] = "full_resync"
        else:
            context["scene_observation_mode"] = "event_delta"
        return context

    def _apply(
        self,
        command: GameCommand,
        decision: LunaDecision,
        planner_model: str,
        command_source: str,
    ) -> tuple[Path, dict[str, object]]:
        action = GameAction[command.action]
        if action.value not in self.observation.available_actions:
            raise ValueError(f"{command.action} is unavailable; allowed={self.observation.available_actions}")

        before = self.current_frame.copy()
        scene_before = self.scene_snapshot
        levels_before = self.observation.levels_completed
        data = {"x": command.x, "y": command.y} if command.action == "ACTION6" else None
        reasoning: dict[str, Any] = {
            "planner_model": planner_model,
            "command_source": command_source,
            "batch_reason": decision.batch_reason,
            "expected_change": decision.expected_change,
            "confidence": decision.confidence,
        }
        observation = self.environment.step(action, data=data, reasoning=reasoning)
        if observation is None:
            raise RuntimeError(f"ARC environment returned no observation for {command.action}")

        self.observation = observation
        self.current_frame = observation.frame[-1].copy()
        self.total_actions += 1
        self.actions_in_level += 1
        self.action_history.append(command.action)
        screenshot = self._capture(command.action)
        level_changed = observation.levels_completed > levels_before
        if level_changed:
            self.object_map.finalize_level(levels_before + 1, self.actions_in_level)
            self.actions_in_level = 0
            self.level_overview_shot = screenshot
            self.last_checked_shot = screenshot
            self.last_planner_decision = None
            self.bfs_state_attempts.clear()
            self.escalation_state_attempts.clear()
            self.semantic_state_visits.clear()
            self.consecutive_stuck_turns = 0
            self.sol_escalations_in_level = 0
            self.last_escalation_planner_turn = -1_000_000
            self.recent_decisions.clear()
            self.action_history.clear()
            self.planner.reset_thread()
            if self.escalation_planner is not None:
                self.escalation_planner.reset_thread()
            self.scene_analyzer.reset_level()
        self._update_scene(self.actions_in_level)
        acknowledge = getattr(self.planner, "acknowledge_transition", None)
        if acknowledge is not None:
            acknowledge(
                before,
                self.current_frame,
                command,
                action_number=self.actions_in_level,
                level=self._current_level_number(),
                available_actions=list(observation.available_actions),
                terminal=observation.state == GameState.WIN,
                reset=observation.state == GameState.GAME_OVER,
            )
        player_ids = self._player_component_ids(decision, scene_before)
        player_motion = self._player_motion(player_ids, self.scene_snapshot)
        important_events = self._important_scene_events(
            scene_before, self.scene_snapshot, player_ids
        )
        ignored_ui_regions = self._ignored_ui_regions(
            scene_before, self.scene_snapshot
        )
        changed_playfield = playfield_changed_pixels(
            before,
            self.current_frame,
            ignored_ui_regions,
        )
        feedback = {
            "action_number": self.total_actions,
            "command": command.model_dump(mode="json"),
            "frame_difference": frame_difference(before, self.current_frame),
            "playfield_changed_pixels": changed_playfield,
            "no_playfield_change": changed_playfield == 0,
            "tracked_player_components": sorted(player_ids),
            "player_motion_detected": player_motion,
            "important_scene_events": important_events,
            "environment_state": observation.state.name,
            "levels_completed": observation.levels_completed,
            "level_changed": level_changed,
            "available_actions": observation.available_actions,
            "screenshot": screenshot.name,
            "perceptual_scene_delta": self._perceptual_scene_delta(),
        }
        self._record({"type": "action", **feedback})
        return screenshot, feedback

    def _recover_game_over(self) -> list[tuple[str, Path]]:
        self.restarts += 1
        if self.restarts > self.max_restarts:
            raise RuntimeError(f"GAME_OVER persisted beyond the {self.max_restarts}-restart limit")
        observation = self.environment.step(GameAction.RESET)
        if observation is None:
            raise RuntimeError("ARC environment returned no observation while recovering GAME_OVER")
        self.observation = observation
        self.current_frame = observation.frame[-1].copy()
        self.actions_in_level = 0
        self.last_feedback = []
        self.last_planner_decision = None
        self.bfs_state_attempts.clear()
        self.escalation_state_attempts.clear()
        self.semantic_state_visits.clear()
        self.consecutive_stuck_turns = 0
        self.sol_escalations_in_level = 0
        self.last_escalation_planner_turn = -1_000_000
        self.recent_decisions.clear()
        self.action_history.clear()
        self.planner.reset_thread()
        if self.escalation_planner is not None:
            self.escalation_planner.reset_thread()
        self.scene_analyzer.reset_level()
        self._update_scene(0)
        shot = self._capture(f"RESET_AFTER_GAME_OVER_{self.restarts}")
        self.level_overview_shot = shot
        self.last_checked_shot = shot
        event = {
            "type": "game_over_recovery",
            "restart": self.restarts,
            "environment_state": observation.state.name,
            "levels_completed": observation.levels_completed,
            "levels_completed_after_reset": observation.levels_completed,
            "screenshot": shot.name,
        }
        self._record(event)
        return [("Fresh reset after GAME_OVER.", shot)]

    def run(self) -> dict[str, object]:
        reset_shot = self._capture("RESET")
        self.level_overview_shot = reset_shot
        self.last_checked_shot = reset_shot
        self._record(
            {
                "type": "reset",
                "game_id": self.observation.game_id,
                "arc_profile_status": self.arc_profile_status,
                "arc_key_source": self.arc_key_source,
                "arc_operation_mode": self.arc_operation_mode,
                "scorecard_id": getattr(self.environment, "scorecard_id", None),
                "available_actions": self.observation.available_actions,
                "environment_state": self.observation.state.name,
                "levels_completed": self.observation.levels_completed,
                "screenshot": reset_shot.name,
                "scene_component_count": self.scene_snapshot["component_count"],
            }
        )
        photos: list[tuple[str, Path]] = [("Reset frame before any movement.", reset_shot)]

        while not self._finished():
            if self.observation.state == GameState.GAME_OVER:
                photos = self._recover_game_over()
                continue
            if self.total_actions >= self.max_actions:
                raise RuntimeError(f"Stopped safely at the {self.max_actions}-action limit")
            if self.planner_turns >= self.max_planner_turns:
                raise RuntimeError(f"Stopped safely at the {self.max_planner_turns}-planner-turn limit")

            allowed = 1 if self.actions_in_level < 3 else self.max_batch
            request_photos: list[tuple[str, Path]] = []
            seen_paths: set[Path] = set()
            resync_period = max(
                1, int(self.visual_feed_policy.get("full_resync_planner_turns", 10))
            )
            full_resync = (
                self.planner.thread_id is None
                or (self.planner_turns + 1) % resync_period == 0
                or not self.visual_feed_policy.get("send_only_latest_between_resyncs", True)
            )
            photo_candidates = []
            if full_resync:
                photo_candidates.append(
                    ("Full-resync overview at the start of this level.", self.level_overview_shot)
                )
            photo_candidates.append(("Current checked frame after the newest actions.", photos[-1][1]))
            for label, path in photo_candidates:
                if path is None:
                    continue
                resolved = path.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                request_photos.append((label, path))

            prior_feedback = list(self.last_feedback)
            recovery_photos: list[tuple[str, Path]] = []
            if self.level_overview_shot is not None:
                recovery_photos.append(
                    ("Level overview for last-resort recovery.", self.level_overview_shot)
                )
            if request_photos[-1][1] != self.level_overview_shot:
                recovery_photos.append(request_photos[-1])

            primary_failed = False
            try:
                result = self.planner.decide(
                    request_photos,
                    self._context(allowed, include_full_scene=full_resync),
                )
            except (RuntimeError, ValueError) as exc:
                failure_event = {
                    "type": "primary_planner_failed",
                    "model": self.planner.model,
                    "planner_turn": self.planner_turns + 1,
                    "error": str(exc),
                }
                self._record(failure_event)
                if self.escalation_planner is None:
                    raise
                emergency_context = self._context(allowed, include_full_scene=True)
                emergency_context.update(
                    {
                        "planner_mode": "last_resort_after_primary_failure",
                        "primary_planner_failure": failure_event,
                    }
                )
                self.sol_consultations += 1
                result = self.escalation_planner.decide_fresh(
                    recovery_photos, emergency_context
                )
                self.sol_escalations += 1
                self.sol_escalations_in_level += 1
                self.last_escalation_planner_turn = self.planner_turns + 1
                primary_failed = True
            self.planner_turns += 1
            primary_decision = result.decision
            primary_source = "sol_emergency" if primary_failed else (
                self.planner.model if getattr(self.planner, "needs_exact_grid", False) else "luna"
            )
            primary_model = (
                self.escalation_planner.model  # type: ignore[union-attr]
                if primary_failed
                else self.planner.model
            )
            primary_thread_id = result.thread_id
            state_id = self._note_primary_decision(primary_decision)
            self.last_checked_shot = photos[-1][1]
            self.last_feedback = []
            bfs_result = self._run_bfs(primary_decision)
            effective_decision = primary_decision
            planner_model = primary_model
            effective_thread_id = primary_thread_id
            selected_by = "sol" if primary_failed else primary_source

            sol_result = None
            if not primary_failed and not (bfs_result and bfs_result.found):
                sol_result = self._run_sol_escalation(
                    decision=primary_decision,
                    bfs_result=bfs_result,
                    state_id=state_id,
                    allowed_commands=allowed,
                    screenshots=recovery_photos,
                    prior_feedback=prior_feedback + list(self.last_feedback),
                )
            if sol_result is not None:
                effective_decision, effective_thread_id, evidence_path = sol_result
                planner_model = self.escalation_planner.model  # type: ignore[union-attr]
                selected_by = "sol"
                self.consecutive_stuck_turns = 0
                print(
                    f"Sol recovery: {evidence_path.name} | "
                    f"confidence={effective_decision.confidence:.2f}",
                    flush=True,
                )
                if effective_decision.request_bfs:
                    sol_bfs_result = self._run_bfs(effective_decision)
                    if sol_bfs_result is not None:
                        bfs_result = sol_bfs_result

            decision = effective_decision
            commands = decision.commands[:allowed]
            if bfs_result and bfs_result.found:
                bfs_batch = bfs_result.actions[: int(self.bfs_policy["execution_batch"])]
                commands = [
                    GameCommand(
                        action=action_name,
                        x=None,
                        y=None,
                        check_after=index == len(bfs_batch) - 1,
                    )
                    for index, action_name in enumerate(bfs_batch)
                ]
                selected_by = "bfs"

            if len(commands) > 1:
                commands = [
                    command.model_copy(update={"check_after": index == len(commands) - 1})
                    for index, command in enumerate(commands)
                ]

            self.object_map.update(
                self._current_level_number(),
                decision.objects,
                self.actions_in_level,
            )
            self.last_planner_decision = {
                "planner_source": selected_by,
                "planner_model": planner_model,
                "thread_id": effective_thread_id,
                **decision.model_dump(mode="json"),
            }
            for index, command in enumerate(commands):
                if command.check_after:
                    commands = commands[: index + 1]
                    break
            self._record(
                {
                    "type": "planner_decision",
                    "planner_turn": self.planner_turns,
                    "source": primary_source,
                    "model": primary_model,
                    "thread_id": primary_thread_id,
                    "usage": result.usage,
                    "allowed_commands": allowed,
                    "proposed_commands": [
                        command.model_dump(mode="json")
                        for command in primary_decision.commands[:allowed]
                    ],
                    "overridden_by": selected_by
                    if selected_by not in {"luna", "sol"}
                    or sol_result is not None
                    else None,
                    "decision": primary_decision.model_dump(mode="json"),
                }
            )

            selection_model = "bounded-bfs" if selected_by == "bfs" else planner_model
            self._record(
                {
                    "type": "planner_selection",
                    "planner_turn": self.planner_turns,
                    "selected_by": selected_by,
                    "model": selection_model,
                    "advising_model": planner_model,
                    "thread_id": effective_thread_id,
                    "confidence": decision.confidence,
                    "goal_status": decision.goal_status,
                    "decision": decision.model_dump(mode="json"),
                    "selected_commands": [
                        command.model_dump(mode="json") for command in commands
                    ],
                }
            )

            names = ",".join(command.action for command in commands)
            print(
                f"Planner turn {self.planner_turns}: {names} | selected_by={selected_by} | "
                f"{decision.goal_status} | confidence={decision.confidence:.2f}",
                flush=True,
            )

            photos = []
            planning_feedback = list(self.last_feedback)
            self.last_feedback = planning_feedback
            for command in commands:
                if self.total_actions >= self.max_actions:
                    break
                try:
                    shot, feedback = self._apply(
                        command, decision, planner_model, selected_by
                    )
                except ValueError as exc:
                    feedback = {"rejected_command": command.model_dump(mode="json"), "reason": str(exc)}
                    self.last_feedback.append(feedback)
                    self._record({"type": "rejected_command", **feedback})
                    break

                self.last_feedback.append(feedback)
                photos.append(
                    (
                        f"Frame after global action {self.total_actions}: {command.action}.",
                        shot,
                    )
                )
                print(
                    f"  step {self.total_actions}: {command.action}, "
                    f"levels={self.observation.levels_completed}/{self.observation.win_levels}, "
                    f"state={self.observation.state.name}",
                    flush=True,
                )
                stop_reason = None
                if feedback["player_motion_detected"] is False:
                    stop_reason = "tracked_player_did_not_move"
                elif feedback["important_scene_events"]:
                    stop_reason = "important_scene_event"
                elif (
                    bool(self.visual_feed_policy.get("stop_batch_on_unchanged_playfield", True))
                    and feedback["no_playfield_change"]
                ):
                    stop_reason = "no_playfield_change_after_action"
                if stop_reason is not None:
                    stop_event = {
                        "type": "batch_stopped",
                        "reason": stop_reason,
                        "action_number": self.total_actions,
                        "command": command.action,
                        "important_scene_events": feedback["important_scene_events"],
                    }
                    self.last_feedback.append(stop_event)
                    self._record(stop_event)
                    break
                if self._finished() or feedback["level_changed"] or command.check_after:
                    break
                if self.observation.state == GameState.GAME_OVER:
                    break
                if self.step_delay:
                    time.sleep(self.step_delay)

            if not photos:
                photos = [("Current frame after a rejected command; choose only an available action.", self._capture("RECHECK"))]

        summary = {
            "game_id": self.observation.game_id,
            "state": self.observation.state.name,
            "levels_completed": self.observation.levels_completed,
            "win_levels": self.observation.win_levels,
            "actions": self.total_actions,
            "planner_turns": self.planner_turns,
            "bfs_attempts": self.bfs_attempts,
            "sol_escalations": self.sol_escalations,
            "sol_consultations": self.sol_consultations,
            "restarts": self.restarts,
            "arc_profile_status": self.arc_profile_status,
            "arc_key_source": self.arc_key_source,
            "arc_operation_mode": self.arc_operation_mode,
            "scorecard_id": getattr(self.environment, "scorecard_id", None),
            "run_directory": str(self.run_dir),
            "object_map_json": str(self.object_map.canonical_json),
            "object_map_markdown": str(self.object_map.canonical_markdown),
            "scene_inventory_json": str(self.scene_json_path),
            "scene_inventory_markdown": str(self.scene_markdown_path),
            "scene_events": str(self.scene_events_path),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UB's Codex CLI ARC screenshot/action loop.")
    parser.add_argument("--game", default="ls20")
    parser.add_argument(
        "--planner",
        default="codex_teacher",
        choices=["codex_teacher", "graph_baseline", "offline_ubx"],
    )
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--search-budget-ms", type=int, default=500)
    parser.add_argument("--max-hypotheses", type=int, default=8)
    parser.add_argument("--disable-neural-expert", action="append", default=[])
    parser.add_argument(
        "--competition",
        action="store_true",
        help="Force offline UB-X planning and ARC competition operation mode.",
    )
    parser.add_argument("--reasoning", default="low", choices=["none", "low", "medium", "high"])
    parser.add_argument("--max-actions", type=int, default=1500)
    parser.add_argument("--max-planner-turns", type=int, default=600)
    parser.add_argument("--max-batch", type=int, default=8, help="Hard cap; Luna normally starts with batches of three.")
    parser.add_argument(
        "--target-levels",
        type=int,
        default=None,
        help="Stop after this many levels; omit to continue until the entire game is won.",
    )
    parser.add_argument("--step-delay", type=float, default=0.0)
    parser.add_argument("--run-root", type=Path, default=WORKSPACE / "ub_runs")
    parser.add_argument("--memory-root", type=Path, default=WORKSPACE / "ub_memory")
    parser.add_argument("--fresh-memory", action="store_true")
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument(
        "--disable-sol-escalation",
        action="store_true",
        help="Disable the gated GPT-5.6 Sol medium recovery planner.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arc_api_key, arc_key_source = load_arc_api_key()
    rules_path = WORKSPACE / "observation_prompt.json"
    planner_name = "offline_ubx" if args.competition else args.planner
    if planner_name == "codex_teacher":
        planner = LunaLightCLIPlanner(
            rules_path,
            model=args.model,
            reasoning_effort=args.reasoning,
            working_directory=WORKSPACE,
            planner_role="primary",
        )
    else:
        planner = OfflinePlannerAdapter(
            rules_path,
            planner=planner_name,
            model_path=args.model_path,
            search_budget_ms=args.search_budget_ms,
            max_hypotheses=args.max_hypotheses,
            disabled_experts=tuple(args.disable_neural_expert),
        )
    escalation_policy = planner.rules["execution_policy"].get("escalation", {})
    escalation_planner = None
    if (
        planner_name == "codex_teacher"
        and escalation_policy.get("enabled", False)
        and not args.disable_sol_escalation
        and not args.competition
    ):
        escalation_planner = LunaLightCLIPlanner(
            rules_path,
            model=str(escalation_policy.get("model", "gpt-5.6-sol")),
            reasoning_effort=str(escalation_policy.get("reasoning_effort", "medium")),
            working_directory=WORKSPACE,
            timeout_seconds=float(escalation_policy.get("timeout_seconds", 240.0)),
            max_attempts=int(planner.rules["execution_policy"].get("planner_retries", 2)),
            planner_role="recovery",
        )
    worker = UBWorker(
        args.game,
        planner,
        escalation_planner=escalation_planner,
        run_root=args.run_root,
        memory_root=args.memory_root,
        max_actions=args.max_actions,
        max_planner_turns=args.max_planner_turns,
        max_batch=args.max_batch,
        target_levels=args.target_levels,
        max_restarts=args.max_restarts,
        fresh_memory=args.fresh_memory,
        arc_api_key=arc_api_key,
        arc_key_source=arc_key_source,
        step_delay=args.step_delay,
        competition=args.competition,
    )
    worker.run()


if __name__ == "__main__":
    main()
