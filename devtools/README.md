# UB Live Viewer (optional)

This is a read-only development aid. It is completely separate from the UB
worker, imports no project modules, and can be removed by deleting `devtools/`.

The normal one-command run loads the existing ARC profile key and starts this
viewer automatically:

```powershell
.\run_ub.ps1 -Game ls20
```

For manual use, open the viewer in another terminal. It automatically switches
to the newest matching run directory:

```powershell
.\.venv\Scripts\python.exe .\devtools\ub_viewer.py --game ls20
```

The window shows the latest step image, current level, action count, planner
turn, active model, confidence, goal status, environment state, the semantic
`object_map.md` table. Raw pixel census data stays hidden by default.

To also show the exhaustive `scene_inventory.md` census:

```powershell
.\.venv\Scripts\python.exe .\devtools\ub_viewer.py --game ls20 --show-census
```

To pin the viewer to one run:

```powershell
.\.venv\Scripts\python.exe .\devtools\ub_viewer.py --run-dir .\ub_runs\ls20_20260720_221849
```

Headless read/parsing check:

```powershell
.\.venv\Scripts\python.exe .\devtools\ub_viewer.py --game ls20 --snapshot
```

The viewer only opens files for brief reads and never locks, imports, or changes
the worker.
