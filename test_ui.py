import os
import sys


if __package__ in {None, ""}:
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    from agent_build.agent1.tui import run_tui
else:
    from agent_build.agent1.tui import run_tui


if __name__ == "__main__":
    raise SystemExit(run_tui())
