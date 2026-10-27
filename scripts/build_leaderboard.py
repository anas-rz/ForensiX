"""Build the public leaderboard from the result files in results/.

    python scripts/build_leaderboard.py

Writes:
- LEADERBOARD.md: the tables, for publishing;
- forensix/server/static/leaderboard.json: the computed data, shipped with the package so a
  server without results/ (e.g. a container) still shows the leaderboard.

The benchmarks, their licenses and reproduce commands are listed in results/leaderboard.json;
all metrics are recomputed from the raw result files by forensix/leaderboard.py.
"""

import argparse
import json
from pathlib import Path

from forensix import leaderboard

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=REPO / "results")
    parser.add_argument("--markdown", type=Path, default=REPO / "LEADERBOARD.md")
    parser.add_argument("--snapshot", type=Path, default=REPO / "forensix" / "server" / "static" / "leaderboard.json")
    args = parser.parse_args()

    board = leaderboard.build(args.results)
    args.markdown.write_text(leaderboard.to_markdown(board))
    args.snapshot.write_text(json.dumps(board, indent=1) + "\n")
    done = [b for b in board["benchmarks"] if b["status"] == "done"]
    print(f"{len(done)} benchmarks, {sum(len(b['rows']) for b in done)} results "
          f"({len(board['benchmarks']) - len(done)} pending) -> {args.markdown}, {args.snapshot}")


if __name__ == "__main__":
    main()
