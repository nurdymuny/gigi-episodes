"""gigi-episodes CLI — change-point detection from the command line.

Usage examples::

    # Detect change-points in a column of a CSV file
    gigi-episodes detect mydata.csv --column latency_ms

    # Output as JSON
    gigi-episodes detect mydata.csv --column latency_ms --json

    # Tune sensitivity
    gigi-episodes detect mydata.csv --column latency_ms --threshold 4.0 --min-segment 20

    # Read from stdin (one number per line)
    cat values.txt | gigi-episodes detect - --column 0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Sequence

from .algorithm import EpisodicResult
from .core import find_changepoints


def _load_column(path: str, column: str) -> List[float]:
    """Load a 1-D column from a CSV file (or stdin if path == '-')."""
    if path == "-":
        reader = csv.reader(sys.stdin)
        rows = list(reader)
    else:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"file not found: {path}")
        with p.open(newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

    if not rows:
        return []

    # Identify column: try as integer index first, then as a header name.
    try:
        col_idx = int(column)
        # Assume no header; just take by index.
        return [float(row[col_idx]) for row in rows if row]
    except ValueError:
        # column is a name — find it in the header row
        header = rows[0]
        if column not in header:
            raise ValueError(
                f"column {column!r} not found. available: {', '.join(header)}"
            )
        col_idx = header.index(column)
        return [float(row[col_idx]) for row in rows[1:] if row]


def _format_text(result: EpisodicResult, source: str, column: str) -> str:
    """Pretty-print an EpisodicResult as text."""
    lines = [
        f"  source:      {source}",
        f"  column:      {column}",
        f"  backend:     {result.backend}",
        f"  threshold:   {result.threshold}",
        f"  min_segment: {result.min_segment}",
        f"  n_points:    {result.n_points}",
        f"  changepoints: {result.count}",
    ]
    if result.count > 0:
        lines.append("")
        lines.append("  index    score   mean_before → mean_after")
        lines.append("  -----    -----   ---------------------------")
        for cp in result.change_points:
            lines.append(
                f"  {cp.index:>5}   {cp.score:6.2f}   {cp.mean_before:8.3f} → {cp.mean_after:8.3f}"
            )
    return "\n".join(lines)


def _format_json(result: EpisodicResult, source: str, column: str) -> str:
    """Render result as JSON."""
    payload = {
        "source": source,
        "column": column,
        "backend": result.backend,
        "threshold": result.threshold,
        "min_segment": result.min_segment,
        "n_points": result.n_points,
        "changepoints": [
            {
                "index": cp.index,
                "score": cp.score,
                "mean_before": cp.mean_before,
                "mean_after": cp.mean_after,
            }
            for cp in result.change_points
        ],
        "count": result.count,
    }
    return json.dumps(payload, indent=2)


def main(argv: Sequence[str] = None) -> int:
    """Entry point for the ``gigi-episodes`` console script."""
    parser = argparse.ArgumentParser(
        prog="gigi-episodes",
        description="Change-point detection in value sequences. Built on GIGI's "
        "EPISODIC brain primitive — see https://davisgeometric.com",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser(
        "detect", help="Detect change-points in a CSV column or stdin"
    )
    detect.add_argument(
        "file", help="Path to CSV file (or '-' for stdin)"
    )
    detect.add_argument(
        "--column", "-c", required=True,
        help="Column name (CSV header) or integer index (0-based)",
    )
    detect.add_argument(
        "--threshold", "-t", type=float, default=3.0,
        help="Number of standard errors above which to flag a change-point. "
        "Higher = stricter. Default: 3.0",
    )
    detect.add_argument(
        "--min-segment", "-m", type=int, default=10,
        help="Minimum segment length before a change-point can be flagged. "
        "Default: 10",
    )
    detect.add_argument(
        "--json", action="store_true",
        help="Output JSON instead of human-readable text",
    )

    args = parser.parse_args(argv)

    if args.command == "detect":
        try:
            values = _load_column(args.file, args.column)
        except (FileNotFoundError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

        result = find_changepoints(
            values, threshold=args.threshold, min_segment=args.min_segment
        )
        out = (
            _format_json(result, args.file, args.column)
            if args.json
            else _format_text(result, args.file, args.column)
        )
        print(out)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
