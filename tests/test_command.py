"""The command builds a whole pack from the real game, and says plainly when it cannot."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from shader.command import BUILT, GAME_JAR, MAC, NOT_BUILT, WINDOWS, PlatformName, RoamingFolder, build, game_folder, requested
from shader.pack import PACK_META, SHADERS, TEXTURES
from shader.autumn import COLOUR_MAPS

# The build's limit, from the requirement.
SECONDS_ALLOWED = 5.0
LIGHT_MAP = "core/lightmap.fsh"


@pytest.fixture(scope="module")
def jar() -> Path:
    """The jar the tests' game comes from, as conftest's fixture chooses it."""
    return Path(os.environ.get("SHADER_GAME_JAR", GAME_JAR))


def test_a_build_writes_the_pack(jar: Path, tmp_path: Path) -> None:
    folder = tmp_path / "pack"
    assert build(requested(jar, folder, [])) == BUILT
    assert (folder / PACK_META).is_file()
    assert (folder / SHADERS / LIGHT_MAP).is_file()
    assert all((folder / TEXTURES / colour_map).is_file() for colour_map in COLOUR_MAPS)


def test_a_missing_game_is_said_plainly_and_nothing_is_written(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "pack"
    assert build(requested(tmp_path / "no-such.jar", folder, [])) == NOT_BUILT
    assert "not written" in capsys.readouterr().out
    assert not folder.exists()


def test_a_folder_that_is_not_our_pack_is_left_alone(jar: Path, tmp_path: Path) -> None:
    folder = tmp_path / "theirs"
    folder.mkdir()
    (folder / "keep.txt").write_text("someone else's")
    assert build(requested(jar, folder, [])) == NOT_BUILT
    assert [entry.name for entry in folder.iterdir()] == ["keep.txt"]


def test_a_build_takes_no_more_than_the_seconds_allowed(jar: Path, tmp_path: Path) -> None:
    # The wait that the requirement limits is the pack maker's, from pressing enter to the pack being
    # there: so the command is run as they run it, in a Python of its own, and the starting of that
    # Python counts. Calling build() here would time it inside the test run instead, where whatever
    # watches the tests slows it many times over: under a tracing tool a 3 second build took 30.
    command = [sys.executable, "-m", "shader.command", "--jar", str(jar), "--pack", str(tmp_path / "pack")]
    started = time.perf_counter()
    finished = subprocess.run(command, capture_output=True, text=True, check=False)
    taken = time.perf_counter() - started
    assert finished.returncode == BUILT, finished.stdout + finished.stderr
    assert taken <= SECONDS_ALLOWED


HOME = Path("/home/player").absolute()
ROAMING = RoamingFolder("/roaming")


def test_the_game_is_looked_for_where_the_launcher_of_each_system_puts_it() -> None:
    assert game_folder(MAC, HOME, None) == HOME / "Library/Application Support/minecraft"
    assert game_folder(PlatformName("linux"), HOME, None) == HOME / ".minecraft"
    assert game_folder(WINDOWS, HOME, ROAMING) == Path(ROAMING) / ".minecraft"


def test_windows_without_a_named_roaming_folder_falls_back_to_the_usual_one() -> None:
    assert game_folder(WINDOWS, HOME, None) == HOME / "AppData/Roaming/.minecraft"
