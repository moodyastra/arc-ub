import arc_agi
from arcengine import GameAction
import sys

from ub_worker import load_arc_api_key


# ARC terminal rendering uses Unicode block glyphs; make it work in Windows shells.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

arc_key, _ = load_arc_api_key()
arc = arc_agi.Arcade(arc_api_key=arc_key)
env = arc.make("ls20", render_mode="terminal")

# Documentation quickstart: issue a few actions and print the scorecard.
for _ in range(10):
    env.step(GameAction.ACTION1)

print(arc.get_scorecard())
