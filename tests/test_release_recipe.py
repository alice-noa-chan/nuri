"""A release plan must forward a deliberate recipe to the correct stage."""

import importlib.util
import json
from pathlib import Path

import pytest


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_release.py"
    spec = importlib.util.spec_from_file_location("release_recipe_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_plan(tmp_path, *, recipe=None, global_recipe=False):
    plan = {"out": str(tmp_path / "runs"), "stages": {
        "base": {"stage": "base", "steps": 4},
        "it": {"stage": "it", "init": "base", "steps": 2},
        "embed": {"stage": "embed", "init": "base", "steps": 2},
    }}
    if global_recipe:
        plan["train"] = {"recipe_config": recipe}
    else:
        plan["stages"]["it"]["recipe_config"] = recipe
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_stage_recipe_is_forwarded_once_without_inheriting_other_stages(tmp_path, monkeypatch):
    path = write_plan(tmp_path, recipe=str(tmp_path / "private-it.json"))
    runner = load_runner()
    commands = []
    monkeypatch.setattr(runner, "run", lambda command, dry_run: commands.append(command) or 0)
    runner.main([str(path), "--dry-run"])
    assert len(commands) == 3
    base, it, embed = commands
    assert "--recipe-config" not in base and "--recipe-config" not in embed
    assert it.count("--recipe-config") == 1
    assert it[it.index("--recipe-config") + 1] == str(tmp_path / "private-it.json")
    assert not (tmp_path / "runs" / "it").exists()


@pytest.mark.parametrize("recipe", [None, "", "  ", [], {}, 4, True])
def test_invalid_recipe_fails_before_subprocess_or_output(tmp_path, monkeypatch, recipe):
    path = write_plan(tmp_path, recipe=recipe)
    runner = load_runner()
    monkeypatch.setattr(runner, "run", lambda *_: pytest.fail("must validate before launch"))
    with pytest.raises(ValueError, match="recipe_config"):
        runner.main([str(path)])
    assert not (tmp_path / "runs").exists()


def test_global_recipe_is_rejected_even_during_dry_run(tmp_path):
    path = write_plan(tmp_path, recipe="private.json", global_recipe=True)
    with pytest.raises(ValueError, match="per stage"):
        load_runner().main([str(path), "--dry-run"])
    assert not (tmp_path / "runs").exists()


def test_legacy_plan_does_not_add_recipe_flags(tmp_path, monkeypatch):
    path = write_plan(tmp_path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan["stages"]["it"].pop("recipe_config")
    path.write_text(json.dumps(plan), encoding="utf-8")
    runner = load_runner()
    commands = []
    monkeypatch.setattr(runner, "run", lambda command, _: commands.append(command) or 0)
    runner.main([str(path), "--dry-run"])
    assert all("--recipe-config" not in command for command in commands)
