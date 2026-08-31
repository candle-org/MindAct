from __future__ import annotations

from pathlib import Path

import pytest

from mindact.cli.main import main

CONFIG = Path(__file__).parents[2] / "configs/experiments/libero-baseline.yaml"


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["--version"])
    assert "mindact 0.1.0" in capsys.readouterr().out


def test_cli_config_check(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config-check", str(CONFIG)]) == 0
    assert capsys.readouterr().out.strip() == "valid: libero-baseline"


def test_cli_fake_eval_supports_overrides(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "eval",
                str(CONFIG),
                "--run-id",
                "cli-run",
                "--output-dir",
                str(tmp_path),
                "--episodes",
                "2",
                "--seed",
                "10",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "evaluated: cli-run" in output
    assert (tmp_path / "cli-run/evaluation/results.json").exists()


def test_cli_fake_eval_rejects_existing_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "cli-run"
    run_dir.mkdir()
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "eval",
                str(CONFIG),
                "--run-id",
                "cli-run",
                "--output-dir",
                str(tmp_path),
            ]
        )
