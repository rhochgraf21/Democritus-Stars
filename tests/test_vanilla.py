"""Reading a game from its client jar: from small jars built here, and from the real one when it is given."""

import io
import json
import os
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from democritus.vanilla import (
    Channel,
    EntryAbsent,
    EntryFormat,
    EntryName,
    EntryUndecodable,
    FieldAbsent,
    FieldKind,
    FormatNumber,
    GameVersion,
    JarPath,
    JarUnreadable,
    Pixel,
    ResourceFormat,
    ShaderFile,
    ShaderPath,
    ShaderText,
    TextureFolder,
    TextureMeta,
    TexturePath,
    VersionField,
    read_game,
    texture_paths,
)

# Where a real client jar can be given, so that the tests also read what Mojang ships; never stored in the repository.
REAL_JAR = "DEMOCRITUS_GAME_JAR"

SHADERS = "assets/minecraft/shaders/"
TEXTURES = "assets/minecraft/textures/"
VERSION = {"id": "26.3", "pack_version": {"resource_major": 97, "resource_minor": 1, "data_major": 121}}
WATER = TexturePath("block/water_still.png")
STONE = TexturePath("block/stone.png")
WATER_META = '{"animation": {"frametime": 2}}'
# Two by two, so that a mix-up of rows and columns shows.
WATER_PIXELS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16))


def png(mode: str, size: tuple[int, int], pixels: list) -> bytes:
    image = Image.new(mode, size)
    image.putdata(pixels)
    stored = io.BytesIO()
    image.save(stored, "PNG")
    return stored.getvalue()


def jar(tmp_path: Path, entries: dict[str, bytes | str]) -> JarPath:
    path = tmp_path / "client.jar"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return JarPath(path)


def game_entries(**replaced: bytes | str | None) -> dict[str, bytes | str]:
    """The entries of a small game; a keyword names an entry by its key below, and None leaves it out."""
    entries = {
        "version": json.dumps(VERSION),
        "core_folder": "",
        "terrain": "#version 330\nvoid main() {}\n",
        "include": "float fog(float d) { return d; }\n",
        "post": "post shader",
        "elsewhere": "not a game shader",
        "water": png("RGBA", (2, 2), list(WATER_PIXELS)),
        "water_meta": WATER_META,
        "stone": png("L", (1, 1), [200]),
        "stone_item": png("L", (1, 1), [0]),
    }
    names = {
        "version": "version.json",
        "core_folder": f"{SHADERS}core/",
        "terrain": f"{SHADERS}core/terrain.vsh",
        "include": f"{SHADERS}include/fog.glsl",
        "post": f"{SHADERS}post/blur.fsh",
        "elsewhere": "com/mojang/shaders/core/other.vsh",
        "water": f"{TEXTURES}{WATER}",
        "water_meta": f"{TEXTURES}{WATER}.mcmeta",
        "stone": f"{TEXTURES}{STONE}",
        "stone_item": f"{TEXTURES}item/stone.png",
    }
    entries.update(replaced)
    return {names[k]: v for k, v in entries.items() if v is not None}


def test_reads_version_and_resource_format(tmp_path: Path) -> None:
    game = read_game(jar(tmp_path, game_entries()), ())
    assert game.version == GameVersion("26.3")
    assert game.resource_format == ResourceFormat(FormatNumber(97), FormatNumber(1))


def test_reads_only_the_core_and_include_shaders_by_their_path_under_shaders(tmp_path: Path) -> None:
    game = read_game(jar(tmp_path, game_entries()), ())
    assert game.shaders == (
        ShaderFile(ShaderPath("core/terrain.vsh"), ShaderText("#version 330\nvoid main() {}\n")),
        ShaderFile(ShaderPath("include/fog.glsl"), ShaderText("float fog(float d) { return d; }\n")),
    )


def test_reads_exactly_the_wanted_textures_in_the_order_wanted(tmp_path: Path) -> None:
    game = read_game(jar(tmp_path, game_entries()), (STONE, WATER))
    assert [t.path for t in game.textures] == [STONE, WATER]


def test_decodes_pixels_row_by_row_from_the_top_left(tmp_path: Path) -> None:
    (water,) = read_game(jar(tmp_path, game_entries()), (WATER,)).textures
    assert (water.width, water.height) == (2, 2)
    assert water.pixels == tuple(Pixel(*map(Channel, p)) for p in WATER_PIXELS)


def test_converts_any_mode_to_rgba(tmp_path: Path) -> None:
    (stone,) = read_game(jar(tmp_path, game_entries()), (STONE,)).textures
    assert stone.pixels == (Pixel(Channel(200), Channel(200), Channel(200), Channel(255)),)


def test_carries_the_animation_beside_a_texture_and_none_where_there_is_none(tmp_path: Path) -> None:
    water, stone = read_game(jar(tmp_path, game_entries()), (WATER, STONE)).textures
    assert water.meta == TextureMeta(WATER_META)
    assert stone.meta is None


def test_names_every_wanted_texture_the_jar_lacks(tmp_path: Path) -> None:
    path = jar(tmp_path, game_entries(water=None))
    lava = TexturePath("block/lava_still.png")
    with pytest.raises(EntryAbsent) as raised:
        read_game(path, (WATER, STONE, lava))
    assert raised.value.jar == path
    assert raised.value.entries == (EntryName(f"{TEXTURES}{WATER}"), EntryName(f"{TEXTURES}{lava}"))
    assert f"{TEXTURES}{lava}" in str(raised.value) and str(path) in str(raised.value)


def test_names_a_missing_version_json(tmp_path: Path) -> None:
    with pytest.raises(EntryAbsent) as raised:
        read_game(jar(tmp_path, game_entries(version=None)), ())
    assert raised.value.entries == (EntryName("version.json"),)


@pytest.mark.parametrize(("version", "field", "kind"), [
    ({"pack_version": VERSION["pack_version"]}, "id", FieldKind.TEXT),
    ({"id": 26, "pack_version": VERSION["pack_version"]}, "id", FieldKind.TEXT),
    ({"id": "26.3"}, "pack_version.resource_major", FieldKind.WHOLE_NUMBER),
    ({"id": "26.3", "pack_version": 97}, "pack_version.resource_major", FieldKind.WHOLE_NUMBER),
    ({"id": "26.3", "pack_version": {"resource_major": 97}}, "pack_version.resource_minor", FieldKind.WHOLE_NUMBER),
    ({"id": "26.3", "pack_version": {"resource_major": "97", "resource_minor": 1}}, "pack_version.resource_major",
     FieldKind.WHOLE_NUMBER),
    ({"id": "26.3", "pack_version": {"resource_major": 97, "resource_minor": True}}, "pack_version.resource_minor",
     FieldKind.WHOLE_NUMBER),
])
def test_names_the_field_version_json_lacks(tmp_path: Path, version: dict, field: str, kind: FieldKind) -> None:
    path = jar(tmp_path, game_entries(version=json.dumps(version)))
    with pytest.raises(FieldAbsent) as raised:
        read_game(path, ())
    assert (raised.value.jar, raised.value.field, raised.value.expected) == (path, VersionField(field), kind)
    assert field in str(raised.value) and "version.json" in str(raised.value)


@pytest.mark.parametrize(("replaced", "entry", "expected"), [
    ({"version": "{"}, "version.json", EntryFormat.JSON),
    ({"version": b"\xff"}, "version.json", EntryFormat.TEXT),
    ({"terrain": b"\xff"}, f"{SHADERS}core/terrain.vsh", EntryFormat.TEXT),
    ({"water": b"not a png"}, f"{TEXTURES}{WATER}", EntryFormat.PNG),
    ({"water_meta": b"\xff"}, f"{TEXTURES}{WATER}.mcmeta", EntryFormat.TEXT),
])
def test_names_an_entry_that_cannot_be_decoded(tmp_path: Path, replaced: dict, entry: str,
                                                expected: EntryFormat) -> None:
    path = jar(tmp_path, game_entries(**replaced))
    with pytest.raises(EntryUndecodable) as raised:
        read_game(path, (WATER,))
    assert (raised.value.jar, raised.value.entry, raised.value.expected) == (path, EntryName(entry), expected)
    assert entry in str(raised.value)


def test_names_a_jar_that_is_not_a_zip_file(tmp_path: Path) -> None:
    path = JarPath(tmp_path / "client.jar")
    path.write_bytes(b"not a zip")
    with pytest.raises(JarUnreadable) as raised:
        read_game(path, ())
    assert raised.value.jar == path and str(path) in str(raised.value)


def test_names_a_jar_that_is_not_there(tmp_path: Path) -> None:
    path = JarPath(tmp_path / "missing.jar")
    with pytest.raises(JarUnreadable) as raised:
        texture_paths(path, TextureFolder("block/"))
    assert raised.value.jar == path


def test_lists_the_textures_under_a_folder_in_order_of_path(tmp_path: Path) -> None:
    path = jar(tmp_path, game_entries())
    assert texture_paths(path, TextureFolder("block/")) == (STONE, WATER)
    assert texture_paths(path, TextureFolder("item/")) == (TexturePath("item/stone.png"),)
    assert texture_paths(path, TextureFolder("entity/")) == ()


# Not linked to the statement: on most runs it is skipped, and a skipped test is no evidence. The small jars above
# are the evidence; this one says whether they still look like what Mojang ships.
@pytest.mark.skipif(REAL_JAR not in os.environ, reason=f"{REAL_JAR} names no client jar")
def test_reads_the_real_jar() -> None:
    path = JarPath(Path(os.environ[REAL_JAR]))
    blocks = texture_paths(path, TextureFolder("block/"))
    assert WATER in blocks
    game = read_game(path, (WATER,))
    assert any(s.path == ShaderPath("core/terrain.vsh") for s in game.shaders)
    assert any(s.path.startswith("include/") for s in game.shaders)
    (water,) = game.textures
    assert len(water.pixels) == water.width * water.height
    assert water.meta is not None
