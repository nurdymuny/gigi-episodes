"""Tests for the gigi_episodes CLI."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gigi_episodes.cli import main


def _make_csv_with_signal(tmpdir: Path) -> Path:
    """Write a CSV with a known-changepoint signal to a temp file. Returns path."""
    rng = np.random.default_rng(7)
    values = np.concatenate([
        rng.normal(0.0, 1.0, 100),
        rng.normal(5.0, 1.0, 100),
    ])
    path = tmpdir / "signal.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "value"])
        for i, v in enumerate(values):
            writer.writerow([i, v])
    return path


def test_cli_detect_text_output(tmp_path, capsys):
    """`gigi-episodes detect file --column value` prints a text report."""
    csv_path = _make_csv_with_signal(tmp_path)
    rc = main(["detect", str(csv_path), "--column", "value"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "changepoints:" in out
    assert "backend:" in out


def test_cli_detect_json_output(tmp_path, capsys):
    """`gigi-episodes detect ... --json` emits valid JSON."""
    csv_path = _make_csv_with_signal(tmp_path)
    rc = main(["detect", str(csv_path), "--column", "value", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert "changepoints" in payload
    assert "count" in payload
    assert payload["backend"] == "local"


def test_cli_detect_by_column_index(tmp_path, capsys):
    """Column can be specified by integer index instead of header name."""
    # Write a CSV without header. Use 200 points (100 per regime) — same scale
    # as the other tests, plenty of room for the windowed comparator to detect.
    path = tmp_path / "noheader.csv"
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 100)])
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        for v in values:
            writer.writerow([v])
    rc = main(["detect", str(path), "--column", "0", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["count"] >= 1


def test_cli_unknown_column_errors(tmp_path, capsys):
    """Specifying a nonexistent column produces an error and nonzero exit."""
    csv_path = _make_csv_with_signal(tmp_path)
    rc = main(["detect", str(csv_path), "--column", "nonexistent"])
    err = capsys.readouterr().err
    assert rc != 0
    assert "nonexistent" in err or "not found" in err


def test_cli_missing_file_errors(capsys):
    """Specifying a nonexistent file produces an error and nonzero exit."""
    rc = main(["detect", "/no/such/file.csv", "--column", "x"])
    err = capsys.readouterr().err
    assert rc != 0
    assert "file" in err.lower() or "not found" in err.lower()
