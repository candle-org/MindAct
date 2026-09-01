from __future__ import annotations

import json
from pathlib import Path

import pytest

import mindact.cli.main as cli_module
from mindact.cli.main import main
from mindact.diagnostics import DiagnosticCheck, DoctorReport
from mindact.experiments import ExperimentManifest

CONFIG = Path(__file__).parents[2] / "configs/experiments/libero-baseline.yaml"


# Keep the command tests independent from optional simulator packages.


class _FakeDoctorReport:
    def render(self) -> str:
        return "fake doctor output"


def make_manifest(path: Path) -> Path:
    manifest_path = path / "manifest.json"
    ExperimentManifest(
        run_id="run-001",
        experiment_name="demo",
        seed=42,
        config={"name": "demo"},
        dataset={"repo_id": "demo/data"},
    ).write_json(manifest_path)
    return manifest_path


def test_cli_doctor(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "run_doctor", lambda: _FakeDoctorReport())

    assert main(["doctor"]) == 0
    assert capsys.readouterr().out.strip() == "fake doctor output"


def test_cli_manifest_show_accepts_run_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = make_manifest(tmp_path / "run-001")

    assert main(["manifest", "show", str(manifest_path.parent)]) == 0
    output = capsys.readouterr().out
    assert "run_id: run-001" in output
    assert 'dataset: {"repo_id": "demo/data"}' in output


def test_cli_manifest_show_rejects_invalid_file(tmp_path: Path) -> None:
    invalid = tmp_path / "manifest.json"
    invalid.write_text("not json")

    with pytest.raises(SystemExit, match="2"):
        main(["manifest", "show", str(invalid)])


def test_doctor_report_is_json_safe() -> None:
    report = DoctorReport(
        checks=(DiagnosticCheck(name="core", status="OK", detail="ready"),)
    )

    assert json.loads(json.dumps(report.to_dict()))["healthy"] is True


def test_cli_manifest_show_accepts_manifest_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = make_manifest(tmp_path / "run-001")

    assert main(["manifest", "show", str(manifest_path)]) == 0
    assert "experiment: demo" in capsys.readouterr().out


def test_cli_doctor_reports_optional_packages_without_importing_them(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_module, "run_doctor", lambda: DoctorReport(
        checks=(
            DiagnosticCheck(name="MindAct", status="OK", detail="0.1.0"),
            DiagnosticCheck(name="LIBERO", status="MISSING", detail="install extra"),
        )
    ))

    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "LIBERO" in output
    assert "MISSING" in output
    assert "install extra" in output


def test_cli_manifest_show_does_not_modify_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = make_manifest(tmp_path / "run-001")
    original = manifest_path.read_bytes()

    main(["manifest", "show", str(manifest_path)])
    capsys.readouterr()

    assert manifest_path.read_bytes() == original



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
