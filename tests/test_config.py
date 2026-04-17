from pathlib import Path

from anghiari.config import Config, load_config


def test_load_config_does_not_create_default_file(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"
    config_file = config_dir / "config.toml"
    monkeypatch.setattr("anghiari.config._CONFIG_DIR", config_dir)
    monkeypatch.setattr("anghiari.config._CONFIG_FILE", config_file)

    cfg = load_config(None, create_default=False)

    assert isinstance(cfg, Config)
    assert not config_file.exists()


def test_load_config_can_create_default_file(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"
    config_file = config_dir / "config.toml"
    monkeypatch.setattr("anghiari.config._CONFIG_DIR", config_dir)
    monkeypatch.setattr("anghiari.config._CONFIG_FILE", config_file)

    cfg = load_config(None, create_default=True)

    assert isinstance(cfg, Config)
    assert config_file.exists()
