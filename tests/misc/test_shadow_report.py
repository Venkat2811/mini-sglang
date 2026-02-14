from __future__ import annotations

import json
import sys

from minisgl.benchmark import shadow_report


def test_shadow_report_summarizes_jsonl(tmp_path, capsys, monkeypatch):
    path = tmp_path / "shadow.jsonl"
    entries = [
        {"kind": "input_mapping", "reason": "value mismatch"},
        {"kind": "input_mapping", "reason": "value mismatch"},
        {"kind": "write_exception", "reason": "forced write failure"},
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["shadow_report", "--input", str(path), "--top", "2"],
    )
    shadow_report.main()
    out = capsys.readouterr().out
    assert "divergence_entries=3" in out
    assert "input_mapping: 2" in out
    assert "write_exception: 1" in out


def test_shadow_report_allow_missing(tmp_path, capsys, monkeypatch):
    missing = tmp_path / "missing.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        ["shadow_report", "--input", str(missing), "--allow-missing"],
    )
    shadow_report.main()
    out = capsys.readouterr().out
    assert "divergence_entries=0" in out
