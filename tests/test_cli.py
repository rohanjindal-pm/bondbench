from pathlib import Path

import pandas as pd

from bondbench.cli import main

P1_RUNS = Path(__file__).resolve().parents[1] / "examples" / "p1-archive" / "runs.csv"


def test_run_offline_writes_outputs(tmp_path, capsys):
    out = tmp_path / "results"
    assert main(["run", "-m", "fake:demo", "-o", str(out), "-q", "--markdown"]) == 0
    runs = pd.read_csv(out / "runs.csv")
    assert len(runs) == 25
    assert (out / "summary.csv").exists()
    assert "`fake:demo`" in (out / "summary.md").read_text()
    assert "| Model |" in capsys.readouterr().out


def test_score_command(tmp_path):
    out = tmp_path / "rescored"
    assert main(["score", str(P1_RUNS), "-o", str(out)]) == 0
    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == 3


def test_validate_builtin(capsys):
    assert main(["validate"]) == 0
    assert "OK   fixed_income_ops_v1: 25 questions" in capsys.readouterr().out


def test_validate_bad_file(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("question\nWhat?\n")
    assert main(["validate", str(bad)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_unknown_model_is_a_clean_error(tmp_path, capsys):
    assert main(["run", "-m", "llama3", "-o", str(tmp_path), "-q"]) == 2
    assert "Can't tell which provider" in capsys.readouterr().err


def test_goldsets_lists_builtin(capsys):
    assert main(["goldsets"]) == 0
    assert "fixed_income_ops_v1: 25 questions" in capsys.readouterr().out
