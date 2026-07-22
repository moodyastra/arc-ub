# UB + Luna Light

UB uses the locally authenticated Codex CLI with `gpt-5.6-luna` and Light
(`low`) reasoning. It does not use API credits. A separate one-shot
`gpt-5.6-sol` Medium planner is available only behind the last-resort gate.

The policy source is `observation_prompt.json`. For every level the worker:

1. captures the reset/level-start frame;
2. asks Luna for one move, applies it, and sends the resulting photo back;
3. repeats that move/check cadence for moves two and three;
4. requests a logical batch of three to eight moves after those probes, stopping
   early only for blocked movement, a discrete visual event, a level change, or completion;
5. inventories every connected color component for diagnostics, then condenses
   it into a small set of whole semantic candidates before Luna sees it;
6. explicitly raises repeated, symmetric, rotated, reflected, scaled, and
   analogous composite candidates above background scenery;
7. sends only the newest checked frame between periodic full visual resyncs;
8. merges the active planner's logical 64x64 object observations into durable JSON and the
   six-column Markdown table;
9. runs bounded BFS only when Luna requests it and, after the first three
   checked moves, actions reach 10, coverage reaches 37%, or Luna's level model
   confidence reaches 85%;
10. executes at most three BFS moves before photographing and consulting Luna
   again;
11. sends a bounded evidence dump to a fresh Sol Medium thread only after a
   failed BFS, two mature stuck checks, or three visits to the same semantic
   state; calls are capped, cooled down, and never made when BFS found a path;
12. starts a compact fresh Luna thread for each new level and continues until
   ARC reports the game won.

Each escalation saves an auditable `sol_escalation_step_*.json` inside that run
directory. Disable the optional fallback with `--disable-sol-escalation`.

The live object memory is available at:

- `ub_memory/<game>/object_map.json` for machines;
- `ub_memory/<game>/object_map.md` for people;
- `ub_runs/<run>/object_map.*` as the exact run snapshot.
- `ub_runs/<run>/scene_inventory.*` as the exhaustive current pixel census;
- `ub_runs/<run>/scene_events.jsonl` as the compact change history.

Run a bounded one-level acceptance test:

```powershell
.\run_ub.ps1 -Game ls20 --target-levels 1 --max-actions 40
```

Run the entire game with the live viewer and the existing ARC profile key from
`.env2.txt` (or `.env2`):

```powershell
.\run_ub.ps1 -Game ls20
```

The launcher never creates or prints a key. It starts the removable viewer,
runs UB in the foreground, and forwards extra worker arguments:

```powershell
.\run_ub.ps1 -Game ls20 --target-levels 1 --max-actions 40
```

Add `-NoViewer` for a submission-style run without the visual aid.

The viewer follows the newest matching run and shows the latest frame, progress,
active model/confidence, and semantic object table. Add `--show-census` when
launching the viewer manually to expose the raw pixel census. Delete `devtools/`
to remove it; the worker never imports it.

Use `--fresh-memory` when you want a completely empty object map rather than
reusing the accessible learned map. Step screenshots, `trace.jsonl`,
`object_map.json`, `object_map.md`, and `summary.json` are written under each
run directory.

## Kaggle boundary

This Codex-CLI version is the local research harness, not yet the final Kaggle
runtime: Kaggle evaluation has no internet access and forces ARC competition
mode. Keep `devtools/` out of the submission, replace the Luna/Sol CLI planners
with an offline packaged planner, and create only one competition scorecard with
one `make()` call per environment.
