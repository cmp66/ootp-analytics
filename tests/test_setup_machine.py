import sys

from scripts import setup_machine


def names(**options):
    return [step.name for step in setup_machine.steps(**options)]


def test_the_steps_run_in_order_and_the_options_add_or_drop_some():
    assert names() == ["check", "aws", "models", "files", "update"]
    assert names(install="cpu", training=True, test=True) == [
        "install", "check", "aws", "models", "files", "training", "update", "test",
    ]  # fmt: skip
    assert "update" not in names(update=False)


def test_the_install_step_is_the_command_of_the_readme_for_the_hardware():
    cpu = setup_machine.steps(install="cpu")[0].command
    assert cpu[:4] == ("uv", "pip", "install", "-r") and "requirements-cpu.txt" in cpu and "unsafe-best-match" in cpu
    assert "download.pytorch.org/whl/cpu" in " ".join(cpu)
    assert setup_machine.steps(install="cuda")[0].command[-1] == "requirements-cuda.txt"
    assert "whl/xpu" in " ".join(setup_machine.steps(install="xpu")[0].command)


def test_the_pull_steps_use_the_sync_scripts_of_this_python():
    by_name = {step.name: step for step in setup_machine.steps(training=True)}
    assert by_name["models"].command[0] == sys.executable and by_name["models"].command[-2:] == (
        str(setup_machine.ROOT / "scripts" / "sync_models.py"),
        "pull",
    )
    assert by_name["training"].command[-3:] == ("pull", "--set", "training")
    assert by_name["check"].command is None and by_name["aws"].command is None


def test_missing_libraries_are_reported_and_a_dry_run_does_nothing(capsys):
    assert setup_machine.missing_modules(("os", "no_such_library")) == ["no_such_library"]
    assert setup_machine.run(setup_machine.steps(install="cpu", test=True), "bucket", dry_run=True) == 0
    assert "Dry run" in capsys.readouterr().out
