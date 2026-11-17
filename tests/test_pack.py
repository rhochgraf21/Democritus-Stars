"""The pack lands where the game looks for it, keeps what the game needs, and never clears a folder not its own."""

import json
from pathlib import Path

import pytest
from PIL import Image

from shader.pack import NotOurPack, PackFolder, write_pack
from shader.vanilla import (
    Channel,
    FormatNumber,
    Pixel,
    PixelCount,
    ResourceFormat,
    ShaderFile,
    ShaderPath,
    ShaderText,
    Texture,
    TextureMeta,
    TexturePath,
)

FORMAT = ResourceFormat(FormatNumber(97), FormatNumber(1))
FOG = ShaderFile(ShaderPath("core/terrain.fsh"), ShaderText("#version 330\n// fog\r\nvoid main() {}\n"))
SKY = ShaderFile(ShaderPath("core/sky.fsh"), ShaderText("#version 330\nvoid main() {}\n"))
WATER = Texture(
    TexturePath("block/water_still.png"),
    PixelCount(2),
    PixelCount(1),
    (Pixel(Channel(10), Channel(20), Channel(30), Channel(40)),
     Pixel(Channel(250), Channel(0), Channel(128), Channel(255))),
    TextureMeta('{"animation": {"frametime": 2}}'),
)
GRASS = Texture(
    TexturePath("colormap/grass.png"),
    PixelCount(1),
    PixelCount(2),
    (Pixel(Channel(1), Channel(2), Channel(3), Channel(4)), Pixel(Channel(5), Channel(6), Channel(7), Channel(8))),
    None,
)


def _folder(tmp_path: Path) -> PackFolder:
    return PackFolder(tmp_path / "resourcepacks" / "shader")


def _files(folder: Path) -> set[Path]:
    return {p.relative_to(folder) for p in folder.rglob("*") if p.is_file()}


def test_shaders_land_where_the_game_looks_byte_for_byte(tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    write_pack(folder, FORMAT, (FOG, SKY), ())
    assert (folder / "assets/minecraft/shaders/core/terrain.fsh").read_bytes() == FOG.text.encode()
    assert (folder / "assets/minecraft/shaders/core/sky.fsh").read_bytes() == SKY.text.encode()


def test_pack_meta_names_the_format(tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    write_pack(folder, FORMAT, (), ())
    pack = json.loads((folder / "pack.mcmeta").read_text())["pack"]
    assert set(pack) == {"description", "min_format", "max_format"}
    assert pack["min_format"] == pack["max_format"] == [97, 1]
    assert pack["description"]


def test_texture_read_back_has_the_same_pixels(tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    write_pack(folder, FORMAT, (), (WATER, GRASS))
    for texture in (WATER, GRASS):
        with Image.open(folder / "assets/minecraft/textures" / texture.path) as image:
            assert image.format == "PNG"
            assert image.mode == "RGBA"
            assert image.size == (texture.width, texture.height)
            assert image.get_flattened_data() == tuple((p.red, p.green, p.blue, p.alpha) for p in texture.pixels)


def test_animation_is_kept_beside_its_texture(tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    write_pack(folder, FORMAT, (), (WATER, GRASS))
    textures = folder / "assets/minecraft/textures"
    assert (textures / "block/water_still.png.mcmeta").read_text() == WATER.meta
    assert not (textures / "colormap/grass.png.mcmeta").exists()


def test_a_rebuild_leaves_nothing_of_the_last_build(tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    write_pack(folder, FORMAT, (FOG, SKY), (WATER,))
    write_pack(folder, FORMAT, (SKY,), (GRASS,))
    shaders, textures = Path("assets/minecraft/shaders"), Path("assets/minecraft/textures")
    assert _files(folder) - {Path("pack.mcmeta"), Path("built-by-shader.txt")} == {
        shaders / "core/sky.fsh", textures / "colormap/grass.png"}


def test_an_empty_folder_is_used(tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    folder.mkdir(parents=True)
    write_pack(folder, FORMAT, (SKY,), ())
    assert (folder / "assets/minecraft/shaders/core/sky.fsh").is_file()


def test_a_folder_not_of_ours_is_refused_and_untouched(tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    folder.mkdir(parents=True)
    (folder / "pack.mcmeta").write_text("theirs")
    (folder / "assets").mkdir()
    (folder / "assets" / "theirs.png").write_bytes(b"theirs")
    with pytest.raises(NotOurPack):
        write_pack(folder, FORMAT, (SKY,), ())
    assert _files(folder) == {Path("pack.mcmeta"), Path("assets/theirs.png")}
    assert (folder / "pack.mcmeta").read_text() == "theirs"


def test_a_file_in_place_of_the_folder_is_refused(tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    folder.parent.mkdir(parents=True)
    folder.write_text("theirs")
    with pytest.raises(NotOurPack):
        write_pack(folder, FORMAT, (SKY,), ())
    assert folder.read_text() == "theirs"
