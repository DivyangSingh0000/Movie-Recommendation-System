import pytest
from pathlib import Path
from automation import automation


def test_example_valid(tmp_path):
    # use the existing example input path from repo if available
    base = Path(__file__).resolve().parents[2]
    example = base / "automation" / "examples" / "input_valid"
    out = tmp_path / "out"
    ok, log = automation.run(example, out)
    assert ok
    assert (out / "summary.json").exists()
