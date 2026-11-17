"""Autumn repaints the game's three colour maps, and reads as autumn."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from zigdag import verifies

from shader.autumn import (
    AUTUMN,
    COLOUR_MAPS,
    DRY_FOLIAGE_MAP,
    FOLIAGE_MAP,
    GRASS_MAP,
    TEMPERATE,
    TROPICAL,
    UNCHANGED,
    Chroma,
    ChromaScale,
    Hue,
    Lightness,
    MissingColourMap,
    Oklch,
    Seasonality,
    Shift,
    Temperature,
    colour_maps,
    oklch,
    partial,
    pixel_of,
    repainted,
    seasonality,
    shifted,
)
from shader.vanilla import Channel, Pixel, PixelCount, Texture, TextureMeta, TexturePath

OPAQUE = Channel(255)

# The foliage colour the game gives a forest, and the grass colour it gives plains.
LEAF_GREEN = Pixel(Channel(89), Channel(174), Channel(48), OPAQUE)
GRASS_GREEN = Pixel(Channel(145), Channel(189), Channel(89), OPAQUE)
LITTER_BROWN = Pixel(Channel(163), Channel(113), Channel(70), OPAQUE)

channels = st.integers(min_value=0, max_value=255).map(Channel)
pixels = st.builds(Pixel, channels, channels, channels, channels)
metas = st.none() | st.text(max_size=8).map(TextureMeta)


@st.composite
def textures(draw: st.DrawFn, path: TexturePath) -> Texture:
    width = draw(st.integers(min_value=1, max_value=5))
    height = draw(st.integers(min_value=0, max_value=3))
    filled = draw(st.lists(pixels, min_size=width * height, max_size=width * height))
    return Texture(path, PixelCount(width), PixelCount(height), tuple(filled), draw(metas))


vanilla_maps = st.tuples(*(textures(path) for path in COLOUR_MAPS))


def flat(path: TexturePath, pixel: Pixel, width: int = 1) -> Texture:
    """A map one row high that holds one colour in every column."""
    return Texture(path, PixelCount(width), PixelCount(1), (pixel,) * width, None)


def one_colour_each(width: int = 1) -> tuple[Texture, ...]:
    return (flat(GRASS_MAP, GRASS_GREEN, width), flat(FOLIAGE_MAP, LEAF_GREEN, width),
            flat(DRY_FOLIAGE_MAP, LITTER_BROWN, width))


def colours_of(maps: tuple[Texture, ...], path: TexturePath) -> tuple[Oklch, ...]:
    by_path = {texture.path: texture for texture in maps}
    return tuple(oklch(pixel) for pixel in by_path[path].pixels)


def painted(path: TexturePath, width: int = 1) -> tuple[Oklch, ...]:
    """What autumn makes of the typical colour of one map, column by column."""
    return colours_of(colour_maps(one_colour_each(width)), path)


def unpainted(path: TexturePath) -> Oklch:
    """The typical colour of one map as the game has it."""
    (colour,) = colours_of(one_colour_each(), path)
    return colour


@verifies("shader.autumn#colours")
def test_the_three_colour_maps_are_the_ones_the_game_tints_with() -> None:
    given = tuple(texture.path for texture in colour_maps(one_colour_each()))
    assert given == COLOUR_MAPS == ("colormap/grass.png", "colormap/foliage.png", "colormap/dry_foliage.png")


@verifies("shader.autumn#colours")
@given(vanilla_maps)
def test_autumn_keeps_the_path_size_alpha_and_meta_of_each_map(vanilla: tuple[Texture, ...]) -> None:
    result = colour_maps(vanilla)
    assert len(result) == len(vanilla)
    for before, after in zip(vanilla, result, strict=True):
        assert (after.path, after.width, after.height, after.meta) == (
            before.path, before.width, before.height, before.meta)
        assert len(after.pixels) == len(before.pixels)
        assert [pixel.alpha for pixel in after.pixels] == [pixel.alpha for pixel in before.pixels]
        assert all(0 <= channel <= 255 and isinstance(channel, int)
                   for pixel in after.pixels for channel in (pixel.red, pixel.green, pixel.blue))


@verifies("shader.autumn#colours")
@given(vanilla_maps, textures(TexturePath("block/stone.png")))
def test_other_textures_and_their_order_change_nothing(vanilla: tuple[Texture, ...], other: Texture) -> None:
    assert colour_maps((other, *reversed(vanilla))) == colour_maps(vanilla)


@verifies("shader.autumn#colours")
@pytest.mark.parametrize("lacking", COLOUR_MAPS)
def test_a_missing_colour_map_is_named(lacking: TexturePath) -> None:
    given_maps = tuple(texture for texture in one_colour_each() if texture.path != lacking)
    with pytest.raises(MissingColourMap, match=lacking):
        colour_maps(given_maps)


@verifies("shader.autumn#colours")
def test_the_first_missing_colour_map_is_the_one_named() -> None:
    with pytest.raises(MissingColourMap) as raised:
        colour_maps(())
    assert str(raised.value) == "the game's textures lack the colour map colormap/grass.png"


@verifies("shader.autumn#colours")
def test_autumn_foliage_is_redder_than_the_games_foliage() -> None:
    autumn = colour_maps(one_colour_each())[1].pixels[0]
    assert autumn.red > LEAF_GREEN.red + 50
    assert autumn.green < LEAF_GREEN.green - 50
    assert autumn.red > autumn.green > autumn.blue


@verifies("shader.autumn#colours")
def test_autumn_grass_is_drier_and_more_olive_than_the_games_grass() -> None:
    (autumn,), game = painted(GRASS_MAP), unpainted(GRASS_MAP)
    assert autumn.chroma < game.chroma
    assert 100 < autumn.hue < game.hue - 15
    assert autumn.lightness < game.lightness


@verifies("shader.autumn#colours")
def test_an_autumn_forest_is_varied_because_warm_climates_turn_less() -> None:
    columns = 20
    across = painted(FOLIAGE_MAP, columns)
    assert across[0] == oklch(LEAF_GREEN)  # the hottest column is a jungle, which has no autumn
    hues = [colour.hue for colour in across]
    assert hues == sorted(hues, reverse=True)  # the colder, the further turned toward red
    assert len(set(hues)) > columns // 3
    assert hues[0] - hues[-1] > 80


@verifies("shader.autumn#colours")
def test_the_hottest_climate_keeps_the_games_colours() -> None:
    columns = 20
    for texture, vanilla in zip(colour_maps(one_colour_each(columns)), one_colour_each(columns), strict=True):
        assert texture.pixels[0] == vanilla.pixels[0]


@verifies("shader.autumn#colours")
def test_each_column_gets_the_share_of_the_shift_that_its_temperature_feels() -> None:
    shift = AUTUMN.foliage
    translucent = Pixel(LEAF_GREEN.red, LEAF_GREEN.green, LEAF_GREEN.blue, Channel(7))
    texture = Texture(FOLIAGE_MAP, PixelCount(4), PixelCount(2), (LEAF_GREEN, GRASS_GREEN, translucent, LEAF_GREEN) * 2,
                      TextureMeta("{}"))
    temperatures = (0.875, 0.625, 0.375, 0.125)  # the middles of four columns, hottest first
    expected = tuple(
        pixel_of(shifted(oklch(pixel), partial(shift, seasonality(Temperature(temperature)))), pixel.alpha)
        for pixel, temperature in zip(texture.pixels[:4], temperatures, strict=True)
    )
    result = repainted(texture, shift)
    assert result.pixels == expected * 2
    assert len({expected[0], expected[2], expected[3]}) == 3
    assert result.pixels[2].alpha == 7


@verifies("shader.autumn#colours")
def test_a_climate_feels_no_season_when_tropical_and_all_of_it_when_temperate() -> None:
    assert seasonality(TROPICAL) == 0.0
    assert seasonality(Temperature(1.0)) == 0.0
    assert seasonality(TEMPERATE) == 1.0
    assert seasonality(Temperature(0.0)) == 1.0
    assert seasonality(Temperature(0.775)) == pytest.approx(0.5)
    assert seasonality(Temperature(0.88)) == pytest.approx(0.2)
    assert (TROPICAL, TEMPERATE) == (0.95, 0.6)


@verifies("shader.autumn#colours")
def test_a_share_of_a_shift_is_that_share_of_each_of_its_parts() -> None:
    shift = Shift(Hue(-90.0), ChromaScale(0.5), Lightness(-0.25))
    half = partial(shift, Seasonality(0.5))
    assert (half.hue_turn, half.chroma_scale, half.lightness_shift) == pytest.approx((-45.0, 0.75, -0.125))
    assert partial(shift, Seasonality(0.0)) == UNCHANGED
    whole = partial(shift, Seasonality(1.0))
    assert (whole.hue_turn, whole.chroma_scale, whole.lightness_shift) == pytest.approx((-90.0, 0.5, -0.25))
    assert UNCHANGED == Shift(Hue(0.0), ChromaScale(1.0), Lightness(0.0))


@verifies("shader.autumn#colours")
def test_a_shift_turns_the_hue_scales_the_chroma_and_raises_the_lightness() -> None:
    colour = Oklch(Lightness(0.5), Chroma(0.25), Hue(30.0))
    moved = shifted(colour, Shift(Hue(-40.0), ChromaScale(0.5), Lightness(0.25)))
    assert (moved.lightness, moved.chroma, moved.hue) == pytest.approx((0.75, 0.125, 350.0))
    assert shifted(colour, Shift(Hue(0.0), ChromaScale(1.0), Lightness(0.75))).lightness == 1.0
    assert shifted(colour, Shift(Hue(0.0), ChromaScale(1.0), Lightness(-0.75))).lightness == 0.0
    assert shifted(colour, Shift(Hue(330.0), ChromaScale(1.0), Lightness(0.0))).hue == pytest.approx(0.0)


@verifies("shader.autumn#colours")
@given(pixels)
def test_a_pixel_comes_back_from_its_colour_unchanged(pixel: Pixel) -> None:
    assert pixel_of(oklch(pixel), pixel.alpha) == pixel


@verifies("shader.autumn#colours")
@pytest.mark.parametrize(
    ("red", "green", "blue", "lightness", "chroma", "hue"),
    [
        # Oklch of the screen's primaries, as published with Oklab.
        (255, 0, 0, 0.62796, 0.25768, 29.234),
        (0, 255, 0, 0.86644, 0.29483, 142.495),
        (0, 0, 255, 0.45201, 0.31321, 264.052),
        (255, 255, 0, 0.96798, 0.21101, 109.769),
        # A pixel dark enough to sit on the straight part of the sRGB curve, worked out by hand.
        (10, 5, 2, 0.12200, 0.01624, 63.80),
    ],
)
def test_the_colours_of_known_pixels(red: int, green: int, blue: int, lightness: float, chroma: float, hue: float) -> None:
    colour = oklch(Pixel(Channel(red), Channel(green), Channel(blue), OPAQUE))
    assert (colour.lightness, colour.chroma) == pytest.approx((lightness, chroma), abs=2e-4)
    assert colour.hue == pytest.approx(hue, abs=0.5)


@verifies("shader.autumn#colours")
def test_greys_have_no_chroma_and_run_from_black_to_white() -> None:
    black, white = (oklch(Pixel(Channel(v), Channel(v), Channel(v), OPAQUE)) for v in (0, 255))
    assert (black.lightness, black.chroma) == (0.0, 0.0)
    assert (white.lightness, white.chroma) == pytest.approx((1.0, 0.0), abs=1e-4)
    assert pixel_of(Oklch(Lightness(1.0), Chroma(0.0), Hue(0.0)), OPAQUE) == Pixel(*(Channel(255),) * 3, OPAQUE)
    assert pixel_of(Oklch(Lightness(0.0), Chroma(0.0), Hue(0.0)), OPAQUE) == Pixel(*(Channel(0),) * 3, OPAQUE)


@verifies("shader.autumn#colours")
@given(
    st.floats(min_value=0.3, max_value=0.9),
    st.floats(min_value=0.35, max_value=1.0),
    st.floats(min_value=0.0, max_value=359.0),
    channels,
)
def test_a_colour_too_strong_for_a_screen_keeps_its_hue_and_lightness(
    lightness: float, chroma: float, hue: float, alpha: Channel
) -> None:
    asked = Oklch(Lightness(lightness), Chroma(chroma), Hue(hue))
    pixel = pixel_of(asked, alpha)
    shown = oklch(pixel)
    assert pixel.alpha == alpha
    assert shown.lightness == pytest.approx(lightness, abs=0.01)
    assert math.cos(math.radians(shown.hue - hue)) > math.cos(math.radians(3.0))
    assert shown.chroma < chroma
    # The strongest colour that shows lies on the edge of what a screen can show.
    assert min(pixel.red, pixel.green, pixel.blue) <= 1 or max(pixel.red, pixel.green, pixel.blue) >= 254
