"""Persistent per-level object memory for UB."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _position_text(position: dict[str, int]) -> str:
    x0, x1 = position["x_min"], position["x_max"]
    y0, y1 = position["y_min"], position["y_max"]
    x_text = str(x0) if x0 == x1 else f"{x0}-{x1}"
    y_text = str(y0) if y0 == y1 else f"{y0}-{y1}"
    return f"({x_text},{y_text})"


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def _is_generic_background(record: dict[str, Any]) -> bool:
    """Reject scenery-only rows while preserving functional route structures."""
    role_words = _tokens(str(record.get("real_life_possible_object", "")))
    background_words = {
        "background",
        "terrain",
        "floor",
        "wall",
        "walls",
        "maze",
        "room",
        "corridor",
        "layout",
        "padding",
        "space",
    }
    functional_words = {
        "barrier",
        "bridge",
        "collectible",
        "door",
        "exit",
        "gate",
        "goal",
        "hazard",
        "key",
        "lock",
        "obstacle",
        "path",
        "player",
        "portal",
        "route",
        "switch",
    }
    return bool(role_words & background_words) and not bool(role_words & functional_words)


def _position_bbox(value: str) -> tuple[int, int, int, int] | None:
    match = re.fullmatch(r"\((\d+)(?:-(\d+))?,(\d+)(?:-(\d+))?\)", value)
    if match is None:
        return None
    x0 = int(match.group(1))
    x1 = int(match.group(2) or x0)
    y0 = int(match.group(3))
    y1 = int(match.group(4) or y0)
    return x0, x1, y0, y1


def _spatially_related(left: str, right: str, tolerance: int = 4) -> bool:
    """Keep stationary identities local; appearance alone is not identity."""
    a = _position_bbox(left)
    b = _position_bbox(right)
    if a is None or b is None:
        return left == right
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    x_gap = max(0, bx0 - ax1, ax0 - bx1)
    y_gap = max(0, by0 - ay1, ay0 - by1)
    return x_gap <= tolerance and y_gap <= tolerance


class ObjectMapStore:
    def __init__(
        self,
        game_id: str,
        run_dir: Path,
        canonical_root: Path,
        *,
        fresh: bool = False,
    ) -> None:
        self.game_id = game_id
        self.run_dir = run_dir
        self.run_json = run_dir / "object_map.json"
        self.run_markdown = run_dir / "object_map.md"
        self.canonical_dir = canonical_root / game_id
        self.canonical_json = self.canonical_dir / "object_map.json"
        self.canonical_markdown = self.canonical_dir / "object_map.md"
        self.canonical_dir.mkdir(parents=True, exist_ok=True)

        if self.canonical_json.exists() and not fresh:
            self.data = json.loads(self.canonical_json.read_text(encoding="utf-8"))
        else:
            self.data = {
                "schema_version": 1,
                "game_id": game_id,
                "revision": 0,
                "updated_at": _now(),
                "levels": {},
            }
        for level in self.data.get("levels", {}).values():
            objects = level.get("objects", {})
            for memory_id in list(objects):
                if _is_generic_background(objects[memory_id]):
                    del objects[memory_id]
        self.data["run_id"] = run_dir.name
        self.save()

    def _level(self, level_number: int) -> dict[str, Any]:
        levels = self.data.setdefault("levels", {})
        key = str(level_number)
        if key not in levels:
            levels[key] = {
                "status": "in_progress",
                "actions_taken": 0,
                "objects": {},
            }
        return levels[key]

    @staticmethod
    def _requested_id_compatible(record: dict[str, Any], candidate: dict[str, Any]) -> bool:
        record_role = _tokens(record["real_life_possible_object"])
        candidate_role = _tokens(candidate["real_life_possible_object"])
        player_words = {"player", "pawn", "avatar", "character"}
        if record_role & player_words and candidate_role & player_words:
            return True
        if _spatially_related(record["position"], candidate["position"], tolerance=4):
            return True
        appearance_score = 0
        if record["color"].lower() == candidate["color"].lower():
            appearance_score += 2
        if {shape.lower() for shape in record["shapes"]} & {
            shape.lower() for shape in candidate["shapes"]
        }:
            appearance_score += 2
        if record_role & candidate_role:
            appearance_score += 2
        return appearance_score >= 6

    @staticmethod
    def _match(
        existing: dict[str, Any],
        candidate: dict[str, Any],
        claimed_ids: set[str] | None = None,
    ) -> str | None:
        claimed_ids = claimed_ids or set()
        analogy_tokens = _tokens(candidate["real_life_possible_object"])
        if analogy_tokens & {"player", "pawn", "avatar", "character"}:
            for memory_id, record in existing.items():
                if memory_id in claimed_ids:
                    continue
                if _tokens(record["real_life_possible_object"]) & {
                    "player",
                    "pawn",
                    "avatar",
                    "character",
                }:
                    return memory_id

        scores: list[tuple[int, str]] = []
        candidate_shapes = {shape.lower() for shape in candidate["shapes"]}
        exact_position_ids = [
            memory_id
            for memory_id, record in existing.items()
            if memory_id not in claimed_ids
            and record["position"] == candidate["position"]
        ]
        if len(exact_position_ids) == 1:
            return exact_position_ids[0]
        for memory_id, record in existing.items():
            if memory_id in claimed_ids:
                continue
            # Non-player objects cannot merge merely because they look alike.
            if not _spatially_related(record["position"], candidate["position"]):
                continue
            score = 0
            if record["position"] == candidate["position"]:
                score += 4
            if record["color"].lower() == candidate["color"].lower():
                score += 2
            if candidate_shapes & {shape.lower() for shape in record["shapes"]}:
                score += 2
            if analogy_tokens & _tokens(record["real_life_possible_object"]):
                score += 2
            scores.append((score, memory_id))
        scores.sort(reverse=True)
        if scores and scores[0][0] >= 6 and (len(scores) == 1 or scores[0][0] > scores[1][0]):
            return scores[0][1]
        return None

    @staticmethod
    def _next_id(level_number: int, existing: dict[str, Any]) -> str:
        prefix = f"L{level_number:02d}-O"
        used = [int(key[len(prefix) :]) for key in existing if key.startswith(prefix) and key[len(prefix) :].isdigit()]
        return f"{prefix}{max(used, default=0) + 1:03d}"

    def update(self, level_number: int, observations: Iterable[Any], action_number: int) -> None:
        level = self._level(level_number)
        objects: dict[str, Any] = level["objects"]
        claimed_ids: set[str] = set()
        for observation in observations:
            item = observation.model_dump(mode="json") if hasattr(observation, "model_dump") else dict(observation)
            candidate = {
                "position": _position_text(item["position"]),
                "color": item["color"],
                "shapes": item["shapes"],
                "real_life_possible_object": item["real_life_possible_object"],
                "importance": item["importance"],
                "speculations": item["speculations"],
                "status": item["status"],
            }
            if _is_generic_background(candidate):
                continue
            requested_id = item.get("memory_id")
            valid_prefix = f"L{level_number:02d}-O"
            valid_requested_id = bool(
                isinstance(requested_id, str)
                and re.fullmatch(rf"{re.escape(valid_prefix)}\d{{3}}", requested_id)
                and requested_id not in claimed_ids
            )
            # A planner may allocate a valid new id after seeing the deterministic
            # scene inventory. Preserve it instead of folding it into a lookalike.
            memory_id = None
            if valid_requested_id:
                existing_record = objects.get(str(requested_id))
                if existing_record is None or self._requested_id_compatible(
                    existing_record, candidate
                ):
                    memory_id = str(requested_id)
            if memory_id is None:
                memory_id = self._match(objects, candidate, claimed_ids)
            if memory_id is None:
                memory_id = self._next_id(level_number, objects)
            claimed_ids.add(memory_id)

            previous = objects.get(memory_id)
            history: list[dict[str, Any]] = []
            first_seen = action_number
            times_seen = 0
            if previous:
                meta = previous.get("_meta", {})
                history = list(meta.get("speculation_history", []))
                first_seen = int(meta.get("first_seen_action", action_number))
                times_seen = int(meta.get("times_seen", 0))
                if previous.get("speculations") != candidate["speculations"]:
                    history.append(
                        {
                            "action": int(meta.get("last_seen_action", action_number)),
                            "speculations": previous.get("speculations", []),
                        }
                    )
                    history = history[-6:]

            objects[memory_id] = {
                **candidate,
                "_meta": {
                    "first_seen_action": first_seen,
                    "last_seen_action": action_number,
                    "times_seen": times_seen + 1,
                    "speculation_history": history,
                },
            }

        level["actions_taken"] = action_number
        level["status"] = "in_progress"
        self.data["revision"] = int(self.data.get("revision", 0)) + 1
        self.data["updated_at"] = _now()
        self.save()

    def finalize_level(self, level_number: int, action_number: int) -> None:
        level = self._level(level_number)
        level["status"] = "complete"
        level["actions_taken"] = action_number
        self.data["revision"] = int(self.data.get("revision", 0)) + 1
        self.data["updated_at"] = _now()
        self.save()

    def compact_level(self, level_number: int, max_rows: int = 12, max_chars: int = 6000) -> str:
        level = self._level(level_number)
        objects = level["objects"]
        ordered = sorted(
            objects.items(),
            key=lambda pair: (
                not bool(_tokens(pair[1]["real_life_possible_object"]) & {"player", "pawn", "avatar"}),
                -int(pair[1]["importance"]),
                -int(pair[1].get("_meta", {}).get("last_seen_action", 0)),
                pair[0],
            ),
        )
        lines = [
            f"MAP L{level_number} rev={self.data.get('revision', 0)} status={level['status']} actions={level['actions_taken']}"
        ]
        for memory_id, record in ordered[:max_rows]:
            speculation = " / ".join(record["speculations"])
            lines.append(
                "|".join(
                    [
                        memory_id,
                        record["position"],
                        record["color"],
                        "+".join(record["shapes"]),
                        record["real_life_possible_object"],
                        str(record["importance"]),
                        record["status"],
                        speculation,
                    ]
                )
            )
        compact = "\n".join(lines)
        return compact[:max_chars]

    def _markdown(self) -> str:
        chunks = [f"# UB object map: {self.game_id}"]
        for level_number in sorted(self.data.get("levels", {}), key=lambda value: int(value)):
            level = self.data["levels"][level_number]
            chunks.extend(
                [
                    "",
                    f"## Level {level_number} - {level['status']}",
                    "",
                    "| Position `(x,y)` | Object color | Shape(s) | Real life possible object | Importance | Two speculations |",
                    "|---|---|---|---|---:|---|",
                ]
            )
            for _, record in sorted(
                level["objects"].items(),
                key=lambda pair: (-int(pair[1]["importance"]), pair[0]),
            ):
                speculation = (
                    f"1. {_safe_cell(record['speculations'][0])}<br>"
                    f"2. {_safe_cell(record['speculations'][1])}"
                )
                chunks.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{_safe_cell(record['position'])}`",
                            _safe_cell(record["color"]),
                            _safe_cell(", ".join(record["shapes"])),
                            _safe_cell(record["real_life_possible_object"]),
                            str(record["importance"]),
                            speculation,
                        ]
                    )
                    + " |"
                )
        return "\n".join(chunks) + "\n"

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)

    def save(self) -> None:
        json_content = json.dumps(self.data, indent=2, ensure_ascii=False) + "\n"
        markdown_content = self._markdown()
        for path, content in (
            (self.canonical_json, json_content),
            (self.canonical_markdown, markdown_content),
            (self.run_json, json_content),
            (self.run_markdown, markdown_content),
        ):
            self._atomic_write(path, content)
