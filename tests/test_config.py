import pytest

import config


def test_env_str_default(monkeypatch):
    monkeypatch.delenv("SOME_STR", raising=False)
    assert config.env_str("SOME_STR", "fallback") == "fallback"


def test_env_str_override(monkeypatch):
    monkeypatch.setenv("SOME_STR", "override")
    assert config.env_str("SOME_STR", "fallback") == "override"


def test_env_int_default(monkeypatch):
    monkeypatch.delenv("SOME_INT", raising=False)
    assert config.env_int("SOME_INT", 4) == 4


def test_env_int_override(monkeypatch):
    monkeypatch.setenv("SOME_INT", "7")
    assert config.env_int("SOME_INT", 4) == 7


def test_env_int_bad_value_exits(monkeypatch):
    monkeypatch.setenv("SOME_INT", "not-a-number")
    with pytest.raises(SystemExit):
        config.env_int("SOME_INT", 4)


def test_env_float_override(monkeypatch):
    monkeypatch.setenv("SOME_FLOAT", "0.5")
    assert config.env_float("SOME_FLOAT", 0.35) == 0.5


def test_env_float_bad_value_exits(monkeypatch):
    monkeypatch.setenv("SOME_FLOAT", "nope")
    with pytest.raises(SystemExit):
        config.env_float("SOME_FLOAT", 0.35)


@pytest.mark.parametrize("raw,expected", [("1", True), ("true", True), ("YES", True), ("on", True)])
def test_env_bool_truthy(monkeypatch, raw, expected):
    monkeypatch.setenv("SOME_BOOL", raw)
    assert config.env_bool("SOME_BOOL") is expected


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "garbage"])
def test_env_bool_falsy(monkeypatch, raw):
    monkeypatch.setenv("SOME_BOOL", raw)
    assert config.env_bool("SOME_BOOL") is False


def test_env_bool_default(monkeypatch):
    monkeypatch.delenv("SOME_BOOL", raising=False)
    assert config.env_bool("SOME_BOOL") is False
    assert config.env_bool("SOME_BOOL", default=True) is True


