import json

import pytest

from local_edit.config import Config, config_path, migrate_legacy_dir


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_EDIT_CONFIG_DIR", str(tmp_path / "cfg"))


def test_defaults_when_no_file_exists():
    cfg = Config.load()
    assert cfg.code_model is None
    assert cfg.recent_models == []


def test_save_then_load_roundtrip():
    cfg = Config.load()
    cfg.code_model = "big-model"
    cfg.fast_model = "small-model"
    cfg.save()

    reloaded = Config.load()
    assert reloaded.code_model == "big-model"
    assert reloaded.fast_model == "small-model"


def test_corrupt_config_falls_back_to_defaults():
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json at all")
    cfg = Config.load()
    assert cfg.code_model is None


def test_unknown_keys_are_ignored():
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"code_model": "m", "some_future_key": 123}))
    cfg = Config.load()
    assert cfg.code_model == "m"


def test_non_dict_config_falls_back():
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2, 3]")
    assert Config.load().code_model is None


def test_remember_model_is_most_recent_first_and_deduped():
    cfg = Config.load()
    cfg.remember_model("a")
    cfg.remember_model("b")
    cfg.remember_model("a")
    assert cfg.recent_models[:2] == ["a", "b"]
    assert cfg.recent_models.count("a") == 1


def test_recent_models_are_capped():
    cfg = Config.load()
    for i in range(30):
        cfg.remember_model(f"m{i}")
    assert len(cfg.recent_models) <= 10


def test_order_models_puts_recent_first_then_alphabetical():
    cfg = Config.load()
    cfg.remember_model("zeta")
    ordered = cfg.order_models(["alpha", "zeta", "beta"])
    assert ordered[0] == "zeta"
    assert ordered[1:] == ["alpha", "beta"]


def test_order_models_ignores_recents_that_are_gone():
    cfg = Config.load()
    cfg.remember_model("unloaded-model")
    ordered = cfg.order_models(["alpha", "beta"])
    assert ordered == ["alpha", "beta"]


def test_save_is_atomic_and_leaves_no_temp_file():
    cfg = Config.load()
    cfg.code_model = "x"
    cfg.save()
    leftovers = list(config_path().parent.glob("*.tmp"))
    assert leftovers == []


def test_legacy_janedit_dir_is_moved(tmp_path):
    (tmp_path / ".janedit").mkdir()
    (tmp_path / ".janedit" / "session.json").write_text("{}")
    migrate_legacy_dir(tmp_path / ".local-edit")
    assert not (tmp_path / ".janedit").exists()
    assert (tmp_path / ".local-edit" / "session.json").read_text() == "{}"


def test_legacy_dir_never_overwrites_new_one(tmp_path):
    (tmp_path / ".janedit").mkdir()
    (tmp_path / ".local-edit").mkdir()
    (tmp_path / ".local-edit" / "keep").write_text("new")
    migrate_legacy_dir(tmp_path / ".local-edit")
    assert (tmp_path / ".janedit").exists()
    assert (tmp_path / ".local-edit" / "keep").read_text() == "new"
