"""Optional, read-only live viewer for UB runs.

This module intentionally has no imports from the UB worker or planner. Delete the
``devtools`` directory to remove it without affecting a submission.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "ub_runs"
DEFAULT_MODEL = "gpt-5.6-luna"
STEP_IMAGE_RE = re.compile(r"^step_(\d+)_.*\.png$", re.IGNORECASE)


@dataclass
class RunStatus:
    """Small, display-oriented summary accumulated from trace events."""

    game: str = "waiting"
    current_level: int = 1
    levels_completed: int = 0
    actions: int = 0
    planner_turn: int = 0
    model: str = DEFAULT_MODEL
    confidence: float | None = None
    goal_status: str = "waiting"
    environment_state: str = "NOT_STARTED"
    latest_screenshot: str | None = None
    last_event_type: str = "waiting"

    def apply(self, event: dict[str, Any], default_model: str = DEFAULT_MODEL) -> None:
        event_type = str(event.get("type", "unknown"))
        self.last_event_type = event_type

        if event_type == "reset":
            game_id = event.get("game_id")
            if game_id:
                self.game = str(game_id).split("-", 1)[0]
            screenshot = event.get("screenshot")
            if screenshot:
                self.latest_screenshot = str(screenshot)

        action_number = _integer(event.get("action_number"))
        if action_number is not None:
            self.actions = max(self.actions, action_number)

        completed = _integer(event.get("levels_completed"))
        if completed is not None:
            self.levels_completed = max(0, completed)

        environment_state = event.get("environment_state") or event.get("state")
        if environment_state:
            self.environment_state = str(environment_state)

        screenshot = event.get("screenshot")
        if screenshot:
            self.latest_screenshot = str(screenshot)

        planner_turn = _integer(
            event.get("planner_turn")
            or event.get("luna_turn")
            or event.get("solver_turn")
        )
        if planner_turn is not None:
            self.planner_turn = max(self.planner_turn, planner_turn)

        decision = event.get("decision")
        if not isinstance(decision, dict):
            decision = {}

        confidence = _number(
            decision.get("confidence")
            if "confidence" in decision
            else event.get("confidence")
        )
        if confidence is not None:
            self.confidence = confidence

        goal_status = decision.get("goal_status") or event.get("goal_status")
        if goal_status:
            self.goal_status = str(goal_status)

        if _is_planner_event(event_type, event):
            self.model = _model_label(event, decision, event_type, default_model)

        won = self.environment_state.upper() in {"WIN", "WON", "COMPLETE", "COMPLETED"}
        self.current_level = max(1, self.levels_completed + (0 if won else 1))


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_planner_event(event_type: str, event: dict[str, Any]) -> bool:
    lowered = event_type.lower()
    return (
        "decision" in lowered
        or "planner" in lowered
        or "escalat" in lowered
        or "consult" in lowered
        or "model" in event
        or "planner_model" in event
    )


def _model_label(
    event: dict[str, Any],
    decision: dict[str, Any],
    event_type: str,
    default_model: str,
) -> str:
    containers: Iterable[dict[str, Any]] = (
        event,
        decision,
        event.get("consultant") if isinstance(event.get("consultant"), dict) else {},
    )
    model: str | None = None
    effort: str | None = None
    for container in containers:
        for key in ("model", "model_name", "planner_model", "solver_model"):
            if container.get(key):
                model = str(container[key])
                break
        for key in ("reasoning_effort", "effort", "model_reasoning_effort"):
            if container.get(key):
                effort = str(container[key])
                break
        if model:
            break

    lowered = event_type.lower()
    if model is None and "sol" in lowered:
        model = "gpt-5.6-sol"
        effort = effort or "medium"
    elif model is None:
        model = default_model

    if effort and effort.lower() not in model.lower():
        return f"{model} ({effort})"
    return model


class TraceReader:
    """Incrementally read complete JSONL records and ignore an unfinished tail."""

    def __init__(self, path: Path, default_model: str = DEFAULT_MODEL) -> None:
        self.path = path
        self.default_model = default_model
        self.status = RunStatus(model=default_model)
        self._offset = 0
        self._carry = b""
        self._identity: tuple[int, int] | None = None

    def reset(self) -> None:
        self.status = RunStatus(model=self.default_model)
        self._offset = 0
        self._carry = b""
        self._identity = None

    def refresh(self) -> RunStatus:
        try:
            stat = self.path.stat()
        except OSError:
            return self.status

        identity = (stat.st_dev, stat.st_ino)
        if (
            self._identity is not None
            and identity != self._identity
            or stat.st_size < self._offset
        ):
            self.reset()
        self._identity = identity

        try:
            with self.path.open("rb") as stream:
                stream.seek(self._offset)
                chunk = stream.read()
                self._offset = stream.tell()
        except OSError:
            return self.status

        if not chunk:
            return self.status

        records = self._carry + chunk
        lines = records.split(b"\n")
        self._carry = lines.pop()
        for raw_line in lines:
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(event, dict):
                self.status.apply(event, self.default_model)
        return self.status


def find_newest_run(runs_root: Path, game: str) -> Path | None:
    """Return the most recently active directory matching ``<game>_*``."""

    try:
        candidates = [
            path
            for path in runs_root.glob(f"{game}_*")
            if path.is_dir()
        ]
    except OSError:
        return None
    if not candidates:
        return None

    def newest_key(path: Path) -> tuple[int, str]:
        try:
            modified = path.stat().st_mtime_ns
        except OSError:
            modified = 0
        return modified, path.name

    return max(candidates, key=newest_key)


def latest_step_image(run_dir: Path, screenshot_hint: str | None = None) -> Path | None:
    """Find the highest numbered complete-looking step image."""

    candidates: list[tuple[int, int, Path]] = []
    try:
        paths = run_dir.glob("step_*.png")
        for path in paths:
            match = STEP_IMAGE_RE.match(path.name)
            if not match or not path.is_file():
                continue
            try:
                modified = path.stat().st_mtime_ns
            except OSError:
                continue
            candidates.append((int(match.group(1)), modified, path))
    except OSError:
        pass

    hinted: Path | None = None
    if screenshot_hint:
        candidate = run_dir / Path(screenshot_hint).name
        if candidate.is_file():
            hinted = candidate

    if not candidates:
        return hinted
    latest = max(candidates, key=lambda item: (item[0], item[1]))[2]
    return latest


def read_text_without_lock(path: Path) -> str | None:
    """Best-effort read suitable for atomically replaced live artifacts."""

    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def resolve_run_dir(args: argparse.Namespace) -> Path | None:
    if args.run_dir is not None:
        return args.run_dir.resolve()
    return find_newest_run(args.runs_root.resolve(), args.game)


def build_snapshot(
    run_dir: Path,
    default_model: str = DEFAULT_MODEL,
    show_census: bool = False,
) -> dict[str, Any]:
    reader = TraceReader(run_dir / "trace.jsonl", default_model)
    status = reader.refresh()
    image_path = latest_step_image(run_dir, status.latest_screenshot)
    map_path = run_dir / "object_map.md"
    map_text = read_text_without_lock(map_path)
    snapshot = {
        "run_dir": str(run_dir),
        "status": asdict(status),
        "latest_screenshot": str(image_path) if image_path else None,
        "object_map": str(map_path) if map_path.exists() else None,
        "object_map_lines": len(map_text.splitlines()) if map_text is not None else 0,
    }
    if show_census:
        scene_path = run_dir / "scene_inventory.md"
        scene_text = read_text_without_lock(scene_path)
        snapshot.update(
            {
                "scene_inventory": str(scene_path) if scene_path.exists() else None,
                "scene_inventory_lines": (
                    len(scene_text.splitlines()) if scene_text is not None else 0
                ),
            }
        )
    return snapshot


class UBViewer:
    """Tk application; imports GUI modules lazily so snapshot mode stays headless."""

    def __init__(self, args: argparse.Namespace) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.args = args
        self.root = tk.Tk()
        self.root.title("UB Live Viewer")
        self.root.geometry("1360x820")
        self.root.minsize(900, 560)
        self.root.configure(bg="#10141c")

        self.run_dir: Path | None = None
        self.trace_reader: TraceReader | None = None
        self._map_signature: tuple[int, ...] | None = None
        self._image_signature: tuple[str, int, int] | None = None
        self._source_image: Any = None
        self._photo: Any = None
        self._resize_job: str | None = None

        self.run_var = tk.StringVar(value="Waiting for a UB run…")
        self.level_var = tk.StringVar(value="1")
        self.actions_var = tk.StringVar(value="0")
        self.turn_var = tk.StringVar(value="0")
        self.model_var = tk.StringVar(value=args.model_label)
        self.confidence_var = tk.StringVar(value="—")
        self.goal_var = tk.StringVar(value="waiting")
        self.state_var = tk.StringVar(value="NOT_STARTED")
        self.updated_var = tk.StringVar(value="Waiting for files")

        self._build_ui()

    def _build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Main.TFrame", background="#10141c")
        style.configure("Panel.TFrame", background="#171d28")
        style.configure(
            "Title.TLabel",
            background="#10141c",
            foreground="#f5f7fa",
            font=("Segoe UI Semibold", 16),
        )
        style.configure(
            "Run.TLabel",
            background="#10141c",
            foreground="#9ba9bd",
            font=("Segoe UI", 9),
        )
        style.configure(
            "CardLabel.TLabel",
            background="#171d28",
            foreground="#8492a6",
            font=("Segoe UI", 8),
        )
        style.configure(
            "CardValue.TLabel",
            background="#171d28",
            foreground="#f5f7fa",
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "PanelTitle.TLabel",
            background="#171d28",
            foreground="#dce5f2",
            font=("Segoe UI Semibold", 11),
        )

        main = ttk.Frame(self.root, style="Main.TFrame", padding=14)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(3, weight=1)

        ttk.Label(main, text="UB Live Viewer", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(main, textvariable=self.run_var, style="Run.TLabel").grid(
            row=1, column=0, sticky="ew", pady=(1, 10)
        )

        cards = ttk.Frame(main, style="Main.TFrame")
        cards.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        card_specs = (
            ("LEVEL", self.level_var),
            ("ACTIONS", self.actions_var),
            ("PLANNER TURN", self.turn_var),
            ("MODEL", self.model_var),
            ("CONFIDENCE", self.confidence_var),
            ("GOAL", self.goal_var),
            ("STATE", self.state_var),
        )
        for index, (label, variable) in enumerate(card_specs):
            cards.columnconfigure(index, weight=2 if label == "MODEL" else 1)
            card = ttk.Frame(cards, style="Panel.TFrame", padding=(10, 7))
            card.grid(row=0, column=index, sticky="nsew", padx=(0, 6))
            ttk.Label(card, text=label, style="CardLabel.TLabel").grid(
                row=0, column=0, sticky="w"
            )
            ttk.Label(card, textvariable=variable, style="CardValue.TLabel").grid(
                row=1, column=0, sticky="w"
            )

        content = ttk.Frame(main, style="Main.TFrame")
        content.grid(row=3, column=0, sticky="nsew")
        content.columnconfigure(0, weight=5)
        content.columnconfigure(1, weight=6)
        content.rowconfigure(0, weight=1)

        image_panel = ttk.Frame(content, style="Panel.TFrame", padding=10)
        image_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        image_panel.columnconfigure(0, weight=1)
        image_panel.rowconfigure(1, weight=1)
        ttk.Label(image_panel, text="Latest frame", style="PanelTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.image_label = tk.Label(
            image_panel,
            text="Waiting for step PNG…",
            bg="#0b0e14",
            fg="#718096",
            font=("Segoe UI", 11),
            anchor="center",
        )
        self.image_label.grid(row=1, column=0, sticky="nsew")
        self.image_label.bind("<Configure>", self._on_image_resize)

        map_panel = ttk.Frame(content, style="Panel.TFrame", padding=10)
        map_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        map_panel.columnconfigure(0, weight=1)
        map_panel.rowconfigure(1, weight=1)
        ttk.Label(
            map_panel,
            text=(
                "Object map + raw scene census"
                if self.args.show_census
                else "Object map"
            ),
            style="PanelTitle.TLabel",
        ).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        map_container = ttk.Frame(map_panel, style="Panel.TFrame")
        map_container.grid(row=1, column=0, sticky="nsew")
        map_container.columnconfigure(0, weight=1)
        map_container.rowconfigure(0, weight=1)
        self.map_text = tk.Text(
            map_container,
            wrap="none",
            bg="#0b0e14",
            fg="#d8e2ef",
            insertbackground="#d8e2ef",
            selectbackground="#315f8c",
            relief="flat",
            padx=10,
            pady=10,
            font=("Cascadia Mono", 9),
            state="disabled",
        )
        map_y = ttk.Scrollbar(map_container, orient="vertical", command=self.map_text.yview)
        map_x = ttk.Scrollbar(map_container, orient="horizontal", command=self.map_text.xview)
        self.map_text.configure(yscrollcommand=map_y.set, xscrollcommand=map_x.set)
        self.map_text.grid(row=0, column=0, sticky="nsew")
        map_y.grid(row=0, column=1, sticky="ns")
        map_x.grid(row=1, column=0, sticky="ew")
        self._set_map_text("Waiting for object_map.md…")

        ttk.Label(main, textvariable=self.updated_var, style="Run.TLabel").grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )

    def run(self) -> None:
        self._poll()
        self.root.mainloop()

    def _poll(self) -> None:
        try:
            candidate = resolve_run_dir(self.args)
            if candidate != self.run_dir:
                self._attach(candidate)
            if self.run_dir is not None and self.trace_reader is not None:
                status = self.trace_reader.refresh()
                self._update_status(status)
                self._update_image(status)
                self._update_map()
                self.updated_var.set(
                    f"Read-only • refreshed {datetime.now().strftime('%H:%M:%S')} • "
                    f"polling every {self.args.poll_ms} ms"
                )
        except Exception as exc:  # Keep a monitoring aid alive on transient file races.
            self.updated_var.set(f"Temporary read error: {exc}")
        finally:
            self.root.after(self.args.poll_ms, self._poll)

    def _attach(self, run_dir: Path | None) -> None:
        self.run_dir = run_dir
        self._map_signature = None
        self._image_signature = None
        self._source_image = None
        self.trace_reader = None
        if run_dir is None:
            self.run_var.set(f"Waiting for {self.args.game}_* under {self.args.runs_root}")
            return
        self.trace_reader = TraceReader(run_dir / "trace.jsonl", self.args.model_label)
        self.run_var.set(str(run_dir))
        self.root.title(f"UB Live Viewer — {run_dir.name}")
        self.image_label.configure(image="", text="Waiting for step PNG…")
        self._set_map_text("Waiting for object_map.md…")

    def _update_status(self, status: RunStatus) -> None:
        self.level_var.set(str(status.current_level))
        self.actions_var.set(str(status.actions))
        self.turn_var.set(str(status.planner_turn))
        self.model_var.set(status.model)
        self.confidence_var.set(
            f"{status.confidence:.0%}" if status.confidence is not None else "—"
        )
        self.goal_var.set(status.goal_status)
        self.state_var.set(status.environment_state)

    def _update_image(self, status: RunStatus) -> None:
        if self.run_dir is None:
            return
        image_path = latest_step_image(self.run_dir, status.latest_screenshot)
        if image_path is None:
            return
        try:
            stat = image_path.stat()
            signature = (str(image_path), stat.st_size, stat.st_mtime_ns)
        except OSError:
            return
        if signature == self._image_signature:
            return

        try:
            from PIL import Image

            # Read into memory first, then close the file immediately. This avoids
            # retaining a handle while the worker creates/replaces later frames.
            image_bytes = image_path.read_bytes()
            with Image.open(io.BytesIO(image_bytes)) as opened:
                source = opened.convert("RGB").copy()
        except (OSError, ValueError):
            return

        self._source_image = source
        self._image_signature = signature
        self._render_image()

    def _on_image_resize(self, _event: Any) -> None:
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(60, self._render_image)

    def _render_image(self) -> None:
        self._resize_job = None
        if self._source_image is None:
            return
        try:
            from PIL import Image, ImageTk

            width = max(1, self.image_label.winfo_width() - 16)
            height = max(1, self.image_label.winfo_height() - 16)
            source_width, source_height = self._source_image.size
            scale = min(width / source_width, height / source_height)
            target = (
                max(1, int(source_width * scale)),
                max(1, int(source_height * scale)),
            )
            rendered = self._source_image.resize(target, Image.Resampling.NEAREST)
            self._photo = ImageTk.PhotoImage(rendered)
            self.image_label.configure(image=self._photo, text="")
        except Exception:
            # A future poll or resize will retry; the viewer must never stop UB.
            return

    def _update_map(self) -> None:
        if self.run_dir is None:
            return
        map_path = self.run_dir / "object_map.md"
        paths = [map_path]
        if self.args.show_census:
            paths.append(self.run_dir / "scene_inventory.md")
        signatures: list[int] = []
        for path in paths:
            try:
                stat = path.stat()
                signatures.extend((stat.st_size, stat.st_mtime_ns))
            except OSError:
                signatures.extend((0, 0))
        signature = tuple(signatures)
        if signature == self._map_signature:
            return
        object_text = read_text_without_lock(map_path)
        if object_text is None:
            return
        self._map_signature = signature
        sections = [object_text.rstrip()]
        if self.args.show_census:
            scene_text = read_text_without_lock(self.run_dir / "scene_inventory.md")
            if scene_text is not None:
                sections.append(scene_text.rstrip())
        self._set_map_text("\n\n".join(sections) + "\n")

    def _set_map_text(self, value: str) -> None:
        self.map_text.configure(state="normal")
        self.map_text.delete("1.0", "end")
        self.map_text.insert("1.0", value)
        self.map_text.configure(state="disabled")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only real-time viewer for UB screenshots, status, and object map."
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--game",
        default="ls20",
        help="Auto-follow the newest ub_runs/<game>_* directory (default: ls20).",
    )
    target.add_argument(
        "--run-dir",
        type=Path,
        help="Follow one specific run directory instead of auto-discovery.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help=f"Run directory root (default: {DEFAULT_RUNS_ROOT}).",
    )
    parser.add_argument(
        "--poll-ms",
        type=int,
        default=250,
        help="Refresh interval in milliseconds (default: 250).",
    )
    parser.add_argument(
        "--model-label",
        default=DEFAULT_MODEL,
        help="Fallback model label for traces that do not record a model.",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Print one JSON status snapshot and exit without starting Tk.",
    )
    parser.add_argument(
        "--show-census",
        action="store_true",
        help="Also display/read the raw scene_inventory.md pixel census.",
    )
    args = parser.parse_args(argv)
    if args.poll_ms < 50:
        parser.error("--poll-ms must be at least 50")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.snapshot:
        run_dir = resolve_run_dir(args)
        if run_dir is None or not run_dir.is_dir():
            print("No matching UB run directory found.", file=sys.stderr)
            return 2
        print(
            json.dumps(
                build_snapshot(run_dir, args.model_label, args.show_census),
                indent=2,
            )
        )
        return 0

    try:
        viewer = UBViewer(args)
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"Viewer dependency unavailable: {exc}", file=sys.stderr)
        return 1
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
