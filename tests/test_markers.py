"""The markers give water, leaves and plants a mark that the shaders can tell apart from every other texture."""

import io
import json
import os
import zipfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image
from zigdag import verifies

from shader.markers import (
    CHANNEL_MAX,
    CHANNEL_MIN,
    MARKED,
    OPAQUE,
    PROBES,
    SIGNATURE_BITS,
    SPRITE_SIDE,
    TRANSPARENT,
    AlphaMark,
    Probe,
    Signature,
    Surface,
    UnexpectedSpriteSize,
    UnmarkedTexture,
    UnreadableFrame,
    mark,
    mark_of,
    surface_of,
)
from shader.vanilla import Channel, Pixel, PixelCount, Texture, TextureMeta, TexturePath

WATER_STILL = TexturePath("block/water_still.png")
OAK_LEAVES = TexturePath("block/oak_leaves.png")
SHORT_GRASS = TexturePath("block/short_grass.png")
SEAGRASS = TexturePath("block/seagrass.png")
STONE = TexturePath("block/stone.png")
GLASS = TexturePath("block/glass.png")
OAK_LOG = TexturePath("block/oak_log.png")
ANIMATION = TextureMeta('{"animation": {"frametime": 2}}')

STEP = 1 << SIGNATURE_BITS
CENTRE = SPRITE_SIDE // 2
LARGEST_SIDE = 4
MOST_FRAMES = 2
HALF_ALPHA = 128

WATER = mark_of(Surface.WATER)
LEAVES = mark_of(Surface.LEAVES)
PLANT = mark_of(Surface.PLANT)

# Where the game keeps its textures inside its jar, and the folder of its block textures there.
JAR_TEXTURES = "assets/minecraft/textures/"
JAR_BLOCKS = JAR_TEXTURES + "block/"
PNG = ".png"
MCMETA = ".mcmeta"
RGBA = "RGBA"

# The game's client jar, such as the one in .minecraft/versions/26.3/26.3.jar.
GAME_JAR = "SHADER_GAME_JAR"

channels = st.integers(CHANNEL_MIN, CHANNEL_MAX).map(Channel)
alphas = st.sampled_from((TRANSPARENT, OPAQUE)) | channels
pixels = st.builds(Pixel, red=channels, green=channels, blue=channels, alpha=alphas)
signed_paths = st.sampled_from(tuple(path for path in MARKED if surface_of(path) is not Surface.WATER))
water_paths = st.sampled_from(tuple(path for path in MARKED if surface_of(path) is Surface.WATER))


def pixel(red: int, green: int, blue: int, alpha: int) -> Pixel:
    return Pixel(Channel(red), Channel(green), Channel(blue), Channel(alpha))


def strip(path: TexturePath, *texels: Pixel, meta: TextureMeta | None = None) -> Texture:
    return Texture(path, PixelCount(len(texels)), PixelCount(1), texels, meta)


def probe_index(probe: Probe, frame: int = 0) -> int:
    return (frame * SPRITE_SIDE + CENTRE + probe.down) * SPRITE_SIDE + CENTRE + probe.across


def sprite(path: TexturePath, fill: Pixel, frames: int = 1, opaque: dict[int, Pixel] | None = None) -> Texture:
    texels = [fill] * (SPRITE_SIDE * SPRITE_SIDE * frames)
    for index, texel in (opaque or {}).items():
        texels[index] = texel
    return Texture(path, SPRITE_SIDE, PixelCount(SPRITE_SIDE * frames), tuple(texels), None)


def low_bits(texel: Pixel) -> tuple[int, int, int]:
    return texel.red % STEP, texel.green % STEP, texel.blue % STEP


def bits_of(signature: Signature) -> tuple[int, int, int]:
    return signature.red, signature.green, signature.blue


def nearest_distance(channel: int, bits: int) -> int:
    return min(abs(value - channel) for value in range(CHANNEL_MIN, CHANNEL_MAX + 1) if value % STEP == bits)


def first_opaque_probe(texture: Texture, frame: int) -> Pixel | None:
    """What the wind effect reads: the first opaque texel among the probes of a frame."""
    probed = (texture.pixels[probe_index(probe, frame)] for probe in PROBES)
    return next((texel for texel in probed if texel.alpha == OPAQUE), None)


@st.composite
def readable_sprites(draw: st.DrawFn) -> Texture:
    frames = draw(st.integers(1, MOST_FRAMES))
    count = SPRITE_SIDE * SPRITE_SIDE * frames
    texels = draw(st.lists(pixels, min_size=count, max_size=count))
    for frame in range(frames):
        index = probe_index(draw(st.sampled_from(PROBES)), frame)
        texels[index] = Pixel(texels[index].red, texels[index].green, texels[index].blue, OPAQUE)
    return Texture(draw(signed_paths), SPRITE_SIDE, PixelCount(count // SPRITE_SIDE), tuple(texels), None)


@st.composite
def water_textures(draw: st.DrawFn) -> Texture:
    width = draw(st.integers(1, LARGEST_SIDE))
    height = draw(st.integers(1, LARGEST_SIDE))
    texels = draw(st.lists(pixels, min_size=width * height, max_size=width * height))
    meta = draw(st.none() | st.just(ANIMATION))
    return Texture(draw(water_paths), PixelCount(width), PixelCount(height), tuple(texels), meta)


@verifies("shader.markers#mark")
def test_water_is_marked_in_alpha_and_leaves_and_plants_in_colour() -> None:
    assert isinstance(mark_of(Surface.WATER), AlphaMark)
    assert isinstance(mark_of(Surface.LEAVES), Signature)
    assert isinstance(mark_of(Surface.PLANT), Signature)


@verifies("shader.markers#mark")
def test_each_surface_has_its_own_mark() -> None:
    leaves, plant = mark_of(Surface.LEAVES), mark_of(Surface.PLANT)
    assert len({mark_of(surface) for surface in Surface}) == len(Surface)
    assert isinstance(leaves, Signature) and isinstance(plant, Signature)
    assert bits_of(leaves) != bits_of(plant)


@verifies("shader.markers#mark")
def test_the_water_mark_is_neither_transparent_nor_opaque() -> None:
    water = mark_of(Surface.WATER)
    assert isinstance(water, AlphaMark)
    assert water.alpha not in (TRANSPARENT, OPAQUE)


@verifies("shader.markers#mark")
def test_no_signature_is_all_zeros_or_all_ones() -> None:
    for surface in (Surface.LEAVES, Surface.PLANT):
        signature = mark_of(surface)
        assert isinstance(signature, Signature)
        assert all(0 <= bit < STEP for bit in bits_of(signature))
        assert bits_of(signature) not in ((0, 0, 0), (STEP - 1, STEP - 1, STEP - 1))


@verifies("shader.markers#mark")
def test_each_probe_alone_makes_a_frame_readable() -> None:
    assert len(set(PROBES)) == len(PROBES)
    for probe in PROBES:
        assert 0 <= CENTRE + probe.across < SPRITE_SIDE
        assert 0 <= CENTRE + probe.down < SPRITE_SIDE
        lone = {probe_index(probe): pixel(30, 60, 90, OPAQUE)}
        (marked,) = mark((sprite(SHORT_GRASS, pixel(0, 0, 0, TRANSPARENT), opaque=lone),))
        seen = first_opaque_probe(marked, 0)
        assert seen is not None
        assert low_bits(seen) == bits_of(PLANT)


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
def test_every_water_texel_takes_the_water_alpha() -> None:
    assert isinstance(WATER, AlphaMark)
    water = strip(WATER_STILL, pixel(10, 20, 30, 180), pixel(40, 50, 60, 0), meta=ANIMATION)
    assert mark((water,)) == (strip(WATER_STILL, pixel(10, 20, 30, WATER.alpha), pixel(40, 50, 60, 0), meta=ANIMATION),)


@verifies("shader.markers#mark")
def test_an_opaque_leaf_texel_moves_to_the_nearest_colour_with_the_signature() -> None:
    assert bits_of(LEAVES) == (4, 6, 4)  # the expected values below are worked out for this signature
    leaves = sprite(OAK_LEAVES, pixel(101, 150, 201, OPAQUE))
    (marked,) = mark((leaves,))
    assert set(marked.pixels) == {pixel(100, 150, 204, OPAQUE)}


@verifies("shader.markers#mark")
def test_a_channel_at_the_end_of_its_range_moves_inward() -> None:
    assert bits_of(PLANT) == (6, 2, 1)  # the expected values below are worked out for this signature
    grass = sprite(SHORT_GRASS, pixel(CHANNEL_MIN, CHANNEL_MAX, CHANNEL_MAX, OPAQUE))
    (marked,) = mark((grass,))
    assert set(marked.pixels) == {pixel(6, 250, 249, OPAQUE)}


@verifies("shader.markers#mark")
def test_texels_that_are_not_opaque_are_left_as_they_are() -> None:
    hole = pixel(1, 2, 3, TRANSPARENT)
    half = pixel(4, 5, 6, HALF_ALPHA)
    opaque = pixel(9, 9, 9, OPAQUE)
    leaves = sprite(OAK_LEAVES, hole, opaque={probe_index(PROBES[0]): opaque, 0: half})
    (marked,) = mark((leaves,))
    assert marked.pixels[0] == half
    assert marked.pixels[1] == hole
    assert marked.pixels[probe_index(PROBES[0])] != opaque


@verifies("shader.markers#mark")
def test_a_frame_with_no_opaque_probe_is_refused() -> None:
    hole = pixel(0, 0, 0, TRANSPARENT)
    readable_first = {probe_index(PROBES[0]): pixel(5, 5, 5, OPAQUE)}
    with pytest.raises(UnreadableFrame) as refused:
        mark((sprite(SEAGRASS, hole, frames=2, opaque=readable_first),))
    assert (refused.value.path, refused.value.frame) == (SEAGRASS, 1)
    assert str(SEAGRASS) in str(refused.value)


@verifies("shader.markers#mark")
def test_an_opaque_texel_beside_the_probes_does_not_make_a_frame_readable() -> None:
    corner = {0: pixel(5, 5, 5, OPAQUE)}
    with pytest.raises(UnreadableFrame):
        mark((sprite(SHORT_GRASS, pixel(0, 0, 0, TRANSPARENT), opaque=corner),))


@verifies("shader.markers#mark")
@pytest.mark.parametrize(("width", "height"), [(32, 32), (16, 24), (8, 16), (16, 0)])
def test_a_leaf_or_plant_texture_of_another_size_is_refused(width: int, height: int) -> None:
    texels = (pixel(5, 5, 5, OPAQUE),) * (width * height)
    with pytest.raises(UnexpectedSpriteSize) as refused:
        mark((Texture(OAK_LEAVES, PixelCount(width), PixelCount(height), texels, None),))
    assert (refused.value.path, refused.value.width, refused.value.height) == (OAK_LEAVES, width, height)
    assert str(OAK_LEAVES) in str(refused.value)


@verifies("shader.markers#mark")
def test_textures_keep_their_order() -> None:
    grass = sprite(SHORT_GRASS, pixel(1, 2, 3, OPAQUE))
    leaves = sprite(OAK_LEAVES, pixel(4, 5, 6, OPAQUE))
    assert [marked.path for marked in mark((grass, leaves))] == [SHORT_GRASS, OAK_LEAVES]
    assert mark(()) == ()


@verifies("shader.markers#mark")
def test_a_texture_that_is_not_marked_is_refused() -> None:
    with pytest.raises(UnmarkedTexture) as refused:
        mark((sprite(OAK_LEAVES, pixel(1, 2, 3, OPAQUE)), strip(STONE, pixel(1, 2, 3, OPAQUE))))
    assert refused.value.path == STONE
    assert str(STONE) in str(refused.value)


@verifies("shader.markers#mark")
@given(readable_sprites())
def test_a_leaf_or_plant_keeps_its_alpha_and_carries_its_signature_in_every_opaque_texel(original: Texture) -> None:
    (marked,) = mark((original,))
    signature = mark_of(surface_of(original.path))
    assert isinstance(signature, Signature)
    assert (marked.path, marked.width, marked.height, marked.meta) == (
        original.path,
        original.width,
        original.height,
        original.meta,
    )
    assert len(marked.pixels) == len(original.pixels)
    for before, after in zip(original.pixels, marked.pixels):
        assert after.alpha == before.alpha
        if before.alpha != OPAQUE:
            assert after == before
            continue
        assert low_bits(after) == bits_of(signature)
        for old, new, bits in zip(
            (before.red, before.green, before.blue), (after.red, after.green, after.blue), bits_of(signature)
        ):
            assert abs(new - old) == nearest_distance(old, bits) < STEP
    for frame in range(original.height // SPRITE_SIDE):
        seen = first_opaque_probe(marked, frame)
        assert seen is not None
        assert low_bits(seen) == bits_of(signature)


@verifies("shader.markers#mark")
@given(water_textures())
def test_water_changes_only_the_alpha_of_texels_that_are_not_transparent(original: Texture) -> None:
    assert isinstance(WATER, AlphaMark)
    (marked,) = mark((original,))
    assert (marked.path, marked.width, marked.height, marked.meta) == (
        original.path,
        original.width,
        original.height,
        original.meta,
    )
    assert len(marked.pixels) == len(original.pixels)
    for before, after in zip(original.pixels, marked.pixels):
        assert (after.red, after.green, after.blue) == (before.red, before.green, before.blue)
        assert after.alpha == (TRANSPARENT if before.alpha == TRANSPARENT else WATER.alpha)


def _texture(jar: zipfile.ZipFile, name: str) -> Texture:
    image = Image.open(io.BytesIO(jar.read(name))).convert(RGBA)
    meta = name + MCMETA
    return Texture(
        path=TexturePath(name.removeprefix(JAR_TEXTURES)),
        width=PixelCount(image.width),
        height=PixelCount(image.height),
        pixels=tuple(Pixel(*(Channel(c) for c in texel)) for texel in image.getdata()),
        meta=TextureMeta(jar.read(meta).decode()) if meta in jar.namelist() else None,
    )


def _block_textures(jar: zipfile.ZipFile) -> tuple[Texture, ...]:
    return tuple(
        _texture(jar, name) for name in jar.namelist() if name.startswith(JAR_BLOCKS) and name.endswith(PNG)
    )


def _cells(texture: Texture) -> tuple[Texture, ...]:
    """Every sprite-sized square of a texture, which is where a wrong guess of "inside" can land."""
    return tuple(
        Texture(
            texture.path,
            SPRITE_SIDE,
            SPRITE_SIDE,
            tuple(
                texture.pixels[(top + row) * texture.width + left + column]
                for row in range(SPRITE_SIDE)
                for column in range(SPRITE_SIDE)
            ),
            None,
        )
        for left in range(0, texture.width - SPRITE_SIDE + 1, SPRITE_SIDE)
        for top in range(0, texture.height - SPRITE_SIDE + 1, SPRITE_SIDE)
    )


# Not linked to shader.markers#mark: in a run without the jar it is skipped, and zigdag reads a linked test
# that ran no code of the markers as linked to the wrong thing.
@pytest.mark.skipif(GAME_JAR not in os.environ, reason=f"{GAME_JAR} names no game jar")
def test_the_game_marks_as_planned_and_no_other_block_texture_carries_a_mark() -> None:
    with zipfile.ZipFile(Path(os.environ[GAME_JAR])) as jar:
        textures = _block_textures(jar)
    marked = tuple(texture for texture in textures if texture.path in MARKED)
    assert {texture.path for texture in marked} == set(MARKED)
    for texture in marked:
        animation = json.loads(texture.meta).get("animation", {}) if texture.meta else {}
        assert not {"width", "height"} & set(animation), f"{texture.path} has frames of its own size"
    mark(marked)

    assert isinstance(WATER, AlphaMark)
    signatures = {bits_of(LEAVES), bits_of(PLANT)}
    false_alphas = []
    false_signatures = []
    for texture in textures:
        surface = surface_of(texture.path)
        if surface is not Surface.WATER and any(texel.alpha == WATER.alpha for texel in texture.pixels):
            false_alphas.append(texture.path)
        if surface in (Surface.LEAVES, Surface.PLANT):
            continue
        for cell in _cells(texture):
            seen = first_opaque_probe(cell, 0)
            if seen is not None and low_bits(seen) in signatures:
                false_signatures.append(texture.path)
    assert false_alphas == []
    assert false_signatures == []
