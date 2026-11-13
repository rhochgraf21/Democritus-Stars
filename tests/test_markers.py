"""The markers give water, leaves and plants an alpha that the shaders can tell apart from every other texture."""

import io
import os
import zipfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image
from zigdag import verifies

from shader.markers import (
    MARKED,
    OPAQUE,
    TRANSPARENT,
    Surface,
    UnmarkedTexture,
    alpha_of,
    mark,
    surface_of,
)
from shader.vanilla import Channel, Pixel, PixelCount, Texture, TextureMeta, TexturePath

WATER_STILL = TexturePath("block/water_still.png")
OAK_LEAVES = TexturePath("block/oak_leaves.png")
SHORT_GRASS = TexturePath("block/short_grass.png")
STONE = TexturePath("block/stone.png")
GLASS = TexturePath("block/glass.png")
OAK_LOG = TexturePath("block/oak_log.png")
ANIMATION = TextureMeta('{"animation": {"frametime": 2}}')

# Where the game keeps its textures inside its jar, and the folder of its block textures there.
JAR_TEXTURES = "assets/minecraft/textures/"
JAR_BLOCKS = JAR_TEXTURES + "block/"
PNG = ".png"
ALPHA_BAND = "A"

# The game's client jar, such as the one in .minecraft/versions/26.3/26.3.jar.
GAME_JAR = "SHADER_GAME_JAR"

LARGEST_SIDE = 4

channels = st.integers(TRANSPARENT, OPAQUE).map(Channel)
pixels = st.builds(Pixel, red=channels, green=channels, blue=channels, alpha=channels)


@st.composite
def textures(draw: st.DrawFn, paths: st.SearchStrategy[TexturePath] = st.sampled_from(MARKED)) -> Texture:
    width = draw(st.integers(1, LARGEST_SIDE))
    height = draw(st.integers(1, LARGEST_SIDE))
    return Texture(
        path=draw(paths),
        width=PixelCount(width),
        height=PixelCount(height),
        pixels=tuple(draw(st.lists(pixels, min_size=width * height, max_size=width * height))),
        meta=draw(st.none() | st.just(ANIMATION)),
    )


def pixel(red: int, green: int, blue: int, alpha: int) -> Pixel:
    return Pixel(Channel(red), Channel(green), Channel(blue), Channel(alpha))


def texture(path: TexturePath, *texels: Pixel, meta: TextureMeta | None = None) -> Texture:
    return Texture(path, PixelCount(len(texels)), PixelCount(1), texels, meta)


@verifies("shader.markers#mark")
def test_each_surface_has_its_own_mark() -> None:
    marks = [alpha_of(surface) for surface in Surface]
    assert len(set(marks)) == len(marks)


@verifies("shader.markers#mark")
def test_no_mark_is_transparent_or_opaque() -> None:
    for surface in Surface:
        assert alpha_of(surface) not in (TRANSPARENT, OPAQUE)


@verifies("shader.markers#mark")
def test_textures_are_told_apart_by_name() -> None:
    assert surface_of(WATER_STILL) is Surface.WATER
    assert surface_of(OAK_LEAVES) is Surface.LEAVES
    assert surface_of(SHORT_GRASS) is Surface.PLANT
    for unmarked in (STONE, GLASS, OAK_LOG):
        assert surface_of(unmarked) is None


@verifies("shader.markers#mark")
def test_every_marked_texture_is_of_a_surface() -> None:
    assert all(surface_of(path) is not None for path in MARKED)
    assert {surface_of(path) for path in MARKED} == set(Surface)


@verifies("shader.markers#mark")
def test_every_water_texel_takes_the_water_mark() -> None:
    water = texture(WATER_STILL, pixel(10, 20, 30, 180), pixel(40, 50, 60, 180), meta=ANIMATION)
    water_mark = alpha_of(Surface.WATER)
    assert mark((water,)) == (
        texture(WATER_STILL, pixel(10, 20, 30, water_mark), pixel(40, 50, 60, water_mark), meta=ANIMATION),
    )


@verifies("shader.markers#mark")
def test_holes_in_leaves_stay_holes() -> None:
    leaves = texture(OAK_LEAVES, pixel(0, 0, 0, 0), pixel(90, 120, 40, 255), pixel(5, 6, 7, 0))
    (marked,) = mark((leaves,))
    assert [texel.alpha for texel in marked.pixels] == [TRANSPARENT, alpha_of(Surface.LEAVES), TRANSPARENT]


@verifies("shader.markers#mark")
def test_textures_keep_their_order() -> None:
    grass = texture(SHORT_GRASS, pixel(1, 2, 3, 255))
    leaves = texture(OAK_LEAVES, pixel(4, 5, 6, 255))
    assert [marked.path for marked in mark((grass, leaves))] == [SHORT_GRASS, OAK_LEAVES]
    assert mark(()) == ()


@verifies("shader.markers#mark")
def test_a_texture_that_is_not_marked_is_refused() -> None:
    with pytest.raises(UnmarkedTexture) as refused:
        mark((texture(OAK_LEAVES, pixel(1, 2, 3, 255)), texture(STONE, pixel(1, 2, 3, 255))))
    assert refused.value.path == STONE
    assert str(STONE) in str(refused.value)


@verifies("shader.markers#mark")
@given(textures())
def test_marking_changes_only_the_alpha_of_texels_that_are_not_transparent(original: Texture) -> None:
    (marked,) = mark((original,))
    surface = surface_of(original.path)
    assert surface is not None
    assert (marked.path, marked.width, marked.height, marked.meta) == (
        original.path,
        original.width,
        original.height,
        original.meta,
    )
    assert len(marked.pixels) == len(original.pixels)
    for before, after in zip(original.pixels, marked.pixels):
        assert (after.red, after.green, after.blue) == (before.red, before.green, before.blue)
        assert after.alpha == (TRANSPARENT if before.alpha == TRANSPARENT else alpha_of(surface))


@verifies("shader.markers#mark")
@given(textures(st.sampled_from((STONE, GLASS, OAK_LOG))))
def test_no_texture_outside_the_marked_is_marked(original: Texture) -> None:
    with pytest.raises(UnmarkedTexture):
        mark((original,))


def _block_alphas(jar: zipfile.ZipFile) -> dict[TexturePath, set[int]]:
    return {
        TexturePath(name.removeprefix(JAR_TEXTURES)): set(
            Image.open(io.BytesIO(jar.read(name))).convert("RGBA").getchannel(ALPHA_BAND).tobytes()
        )
        for name in jar.namelist()
        if name.startswith(JAR_BLOCKS) and name.endswith(PNG)
    }


# Not linked to shader.markers#mark: in a run without the jar it is skipped, and zigdag reads a linked test
# that ran no code of the markers as linked to the wrong thing.
@pytest.mark.skipif(GAME_JAR not in os.environ, reason=f"{GAME_JAR} names no game jar")
def test_the_game_holds_every_marked_texture_and_no_other_holds_a_mark() -> None:
    with zipfile.ZipFile(Path(os.environ[GAME_JAR])) as jar:
        alphas = _block_alphas(jar)
    marks = {alpha_of(surface) for surface in Surface}
    assert set(MARKED) <= set(alphas)
    false_marks = {
        path: sorted(held & marks) for path, held in alphas.items() if path not in MARKED and held & marks
    }
    assert false_marks == {}
