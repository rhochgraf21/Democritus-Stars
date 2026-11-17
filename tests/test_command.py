"""The command builds a whole pack from the real game, and says plainly when it cannot."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from zigdag import verifies

from shader.cloudmap import CLOUD_MAP
from shader.command import BUILT, NOT_BUILT, GAME_JAR, build, requested
from shader.pack import PACK_META, SHADERS, TEXTURES
from shader.seasons import COLOUR_MAPS

# The build's limit, from the requirement.
SECONDS_ALLOWED = 5.0
LIGHT_MAP = "core/lightmap.fsh"


@pytest.fixture(scope="module")
def jar() -> Path:
    """The jar the tests' game comes from, as conftest's fixture chooses it."""
    return Path(os.environ.get("SHADER_GAME_JAR", GAME_JAR))


@verifies("shader.command#run")
def test_a_build_writes_the_pack_for_the_season(jar: Path, tmp_path: Path) -> None:
    folder = tmp_path / "pack"
    assert build(requested("autumn", jar, folder, [])) == BUILT
    assert (folder / PACK_META).is_file()
    assert (folder / SHADERS / LIGHT_MAP).is_file()
    assert all((folder / TEXTURES / colour_map).is_file() for colour_map in COLOUR_MAPS)


@verifies("shader.command#run")
def test_the_filled_cloud_map_goes_only_where_the_clouds_effect_goes(jar: Path, tmp_path: Path) -> None:
    # Under the game's own cloud shaders the filled map is one slab of cloud over the whole sky.
    with_clouds, without = tmp_path / "with", tmp_path / "without"
    assert build(requested("summer", jar, with_clouds, [])) == BUILT
    assert build(requested("summer", jar, without, ["clouds"])) == BUILT
    assert (with_clouds / TEXTURES / CLOUD_MAP).is_file()
    assert not (without / TEXTURES / CLOUD_MAP).exists()


@verifies("shader.command#run")
def test_another_season_gives_other_colours(jar: Path, tmp_path: Path) -> None:
    build(requested("autumn", jar, tmp_path / "autumn", []))
    build(requested("winter", jar, tmp_path / "winter", []))
    grass = TEXTURES / COLOUR_MAPS[0]
    assert (tmp_path / "autumn" / grass).read_bytes() != (tmp_path / "winter" / grass).read_bytes()


@verifies("shader.command#run")
def test_a_missing_game_is_said_plainly_and_nothing_is_written(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "pack"
    assert build(requested("autumn", tmp_path / "no-such.jar", folder, [])) == NOT_BUILT
    assert "not written" in capsys.readouterr().out
    assert not folder.exists()


@verifies("shader.command#run")
def test_a_folder_that_is_not_our_pack_is_left_alone(jar: Path, tmp_path: Path) -> None:
    folder = tmp_path / "theirs"
    folder.mkdir()
    (folder / "keep.txt").write_text("someone else's")
    assert build(requested("autumn", jar, folder, [])) == NOT_BUILT
    assert [entry.name for entry in folder.iterdir()] == ["keep.txt"]


@verifies("shader.command#quick")
def test_a_build_takes_no_more_than_the_seconds_allowed(jar: Path, tmp_path: Path) -> None:
    # The wait that the requirement limits is the pack maker's, from pressing enter to the pack being
    # there: so the command is run as they run it, in a Python of its own, and the starting of that
    # Python counts. Calling build() here would time it inside the test run instead, where whatever
    # watches the tests slows it many times over: zigdag's watching made a 3 second build take 30.
    command = [sys.executable, "-m", "shader.command", "--season", "autumn", "--jar", str(jar), "--pack", str(tmp_path / "pack")]
    started = time.perf_counter()
    finished = subprocess.run(command, capture_output=True, text=True, check=False)
    taken = time.perf_counter() - started
    assert finished.returncode == BUILT, finished.stdout + finished.stderr
    assert taken <= SECONDS_ALLOWED
