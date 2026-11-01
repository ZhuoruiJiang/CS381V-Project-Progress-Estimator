from datetime import datetime

import pytest

from coffee_progress.evaluation.cli import output_directory

NOW = datetime(2026, 11, 1, 10, 20, 30)


@pytest.mark.parametrize("mode", ["qualitative", "offline"])
def test_named_and_unnamed_output_paths(mode):
    assert output_directory(mode, "dummy", now=NOW).as_posix() == (
        "outputs/evaluations/dummy_2026-11-01-10-20-30"
    )
    assert output_directory(mode, now=NOW).as_posix() == ("outputs/evaluations/2026-11-01-10-20-30")
    assert output_directory(mode, "experiment/model", now=NOW).as_posix() == (
        "outputs/evaluations/experiment/model_2026-11-01-10-20-30"
    )


@pytest.mark.parametrize(
    "name", ["../outside", "/absolute", "C:\\absolute", "C:relative", ".", "bad:name"]
)
def test_output_cannot_escape_mode_folder(name):
    with pytest.raises(ValueError, match="relative run name"):
        output_directory("offline", name, now=NOW)


@pytest.mark.parametrize("mode", ["qualitative", "offline", None])
def test_cli_accepts_omitted_output_and_passes_generated_directory(mode, monkeypatch, tmp_path):
    from coffee_progress import DatasetConfig
    from coffee_progress.evaluation import cli

    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("seed: 0\n")
    expected = tmp_path / "generated_run"
    routed, calls = [], []

    def resolve(selected_mode, name):
        routed.append((selected_mode, name))
        return expected

    def evaluate(estimator, dataset, directory, **kwargs):
        calls.append(directory)

    monkeypatch.setattr(cli, "load_config", lambda path: DatasetConfig())
    monkeypatch.setattr(cli, "ProgressDataset", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "load_factory", lambda spec: lambda **kwargs: object())
    monkeypatch.setattr(cli, "output_directory", resolve)
    monkeypatch.setattr(cli, "evaluate_qualitative", evaluate)
    monkeypatch.setattr(cli, "evaluate_offline", evaluate)
    monkeypatch.setattr(
        "sys.argv",
        [
            "coffee-evaluate",
            *(["--mode", mode] if mode is not None else []),
            "--estimator",
            "example:create",
            "--protocol",
            str(protocol),
        ],
    )
    cli.main()
    effective_mode = mode or "offline"
    assert routed == [(effective_mode, None)]
    assert calls == [expected if effective_mode == "offline" else expected / "qualitative"]
