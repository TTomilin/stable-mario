import os

import pytest

import stable_retro


@pytest.fixture
def custom_cleanup():
    stable_retro.data.Integrations.clear_custom_paths()
    assert not stable_retro.data.Integrations.CUSTOM_ONLY.paths

    yield

    stable_retro.data.Integrations.clear_custom_paths()
    assert not stable_retro.data.Integrations.CUSTOM_ONLY.paths


def test_basic_paths():
    assert stable_retro.data.Integrations.STABLE.paths == ["stable"]
    assert stable_retro.data.Integrations.CONTRIB_ONLY.paths == ["contrib"]
    assert stable_retro.data.Integrations.EXPERIMENTAL_ONLY.paths == ["experimental"]
    assert not stable_retro.data.Integrations.CUSTOM_ONLY.paths

    assert stable_retro.data.Integrations.CONTRIB.paths == ["contrib", "stable"]
    assert stable_retro.data.Integrations.EXPERIMENTAL.paths == ["experimental", "stable"]
    assert stable_retro.data.Integrations.CUSTOM.paths == ["stable"]

    assert stable_retro.data.Integrations.ALL.paths == ["contrib", "experimental", "stable"]


def test_custom_path(custom_cleanup):
    assert not stable_retro.data.Integrations.CUSTOM_ONLY.paths
    assert stable_retro.data.Integrations.CUSTOM.paths == ["stable"]

    stable_retro.data.Integrations.add_custom_path("a")
    assert stable_retro.data.Integrations.CUSTOM_ONLY.paths == ["a"]
    assert stable_retro.data.Integrations.CUSTOM.paths == ["a", "stable"]

    stable_retro.data.Integrations.add_custom_path("b")
    assert stable_retro.data.Integrations.CUSTOM_ONLY.paths == ["a", "b"]
    assert stable_retro.data.Integrations.CUSTOM.paths == ["a", "b", "stable"]


def test_custom_path_default(custom_cleanup):
    assert not stable_retro.data.Integrations.CUSTOM_ONLY.paths
    assert stable_retro.data.Integrations.CUSTOM.paths == ["stable"]
    assert stable_retro.data.Integrations.DEFAULT.paths == ["stable"]

    stable_retro.data.add_custom_integration("a")
    assert stable_retro.data.Integrations.CUSTOM_ONLY.paths == ["a"]
    assert stable_retro.data.Integrations.CUSTOM.paths == ["a", "stable"]
    assert stable_retro.data.Integrations.DEFAULT.paths == ["a", "stable"]

    stable_retro.data.DefaultIntegrations.reset()
    assert stable_retro.data.Integrations.CUSTOM_ONLY.paths == ["a"]
    assert stable_retro.data.Integrations.CUSTOM.paths == ["a", "stable"]
    assert stable_retro.data.Integrations.DEFAULT.paths == ["stable"]


def test_custom_path_absolute(custom_cleanup):
    assert not stable_retro.data.get_file_path(
        "",
        "Dekadence-Dekadrive.md",
        inttype=stable_retro.data.Integrations.CUSTOM_ONLY,
    )

    test_rom_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../roms")
    stable_retro.data.Integrations.add_custom_path(test_rom_dir)
    assert stable_retro.data.get_file_path(
        "",
        "Dekadence-Dekadrive.md",
        inttype=stable_retro.data.Integrations.CUSTOM_ONLY,
    ) == os.path.join(test_rom_dir, "Dekadence-Dekadrive.md")


def test_custom_path_relative(custom_cleanup):
    assert not stable_retro.data.get_file_path(
        "Airstriker-Genesis",
        "rom.md",
        inttype=stable_retro.data.Integrations.CUSTOM_ONLY,
    )

    stable_retro.data.Integrations.add_custom_path(stable_retro.data.Integrations.STABLE.paths[0])
    assert stable_retro.data.get_file_path(
        "Airstriker-Genesis",
        "rom.md",
        inttype=stable_retro.data.Integrations.CUSTOM_ONLY,
    ) == stable_retro.data.get_file_path(
        "Airstriker-Genesis",
        "rom.md",
        inttype=stable_retro.data.Integrations.STABLE,
    )
