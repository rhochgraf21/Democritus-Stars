"""The sky moves from the fog colour at the horizon to the sky colour overhead, and fits the real game."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from shader.composer import compose
from shader.effects.sky import (
    ALL_FOG,
    FULL_LIGHT,
    GRADIENT_FUNCTION,
    GRADIENT_INCLUDE_LINE,
    GRADIENT_INCLUDE_PATH,
    HAZE_REACH,
    HORIZON_HEIGHT,
    NO_FOG,
    NO_LIGHT,
    ROUNDING,
    STRAIGHT_DOWN,
    STRAIGHT_UP,
    Colour,
    Height,
    Intensity,
    effect,
    fog_share,
    gradient_include,
    sky_colour,
)
from shader.validator import validate
from shader.vanilla import Game, ShaderPath, ShaderText

SKY_VERTEX = ShaderPath("core/sky.vsh")
SKY_FRAGMENT = ShaderPath("core/sky.fsh")

NOON_SKY = Colour(Intensity(0.47), Intensity(0.655), Intensity(1.0))
NOON_FOG = Colour(Intensity(0.753), Intensity(0.847), Intensity(1.0))

# The photograph the look was chosen against: most of the change is over this far up.
MOSTLY_SKY_BY = Height(math.sin(math.radians(25.0)))
MOST = 0.75
# And the first few degrees above the horizon still read as the horizon.
STILL_HORIZON_AT = Height(math.sin(math.radians(5.0)))
NEARLY_ALL = 0.9

intensities = st.floats(NO_LIGHT, FULL_LIGHT).map(Intensity)
colours = st.builds(Colour, intensities, intensities, intensities)
heights = st.floats(STRAIGHT_DOWN, STRAIGHT_UP).map(Height)
below_the_horizon = st.floats(STRAIGHT_DOWN, HORIZON_HEIGHT).map(Height)


def rise_to_height(rise: float) -> Height:
    """The height that lies this share of the way from the horizon to straight up."""
    return Height(HORIZON_HEIGHT + rise * (STRAIGHT_UP - HORIZON_HEIGHT))


def channels(colour: Colour) -> tuple[Intensity, Intensity, Intensity]:
    return (colour.red, colour.green, colour.blue)


def distance(one: Colour, other: Colour) -> float:
    return sum(abs(a - b) for a, b in zip(channels(one), channels(other), strict=True))


@given(colours, colours, below_the_horizon)
def test_at_the_horizon_and_below_the_sky_is_the_fog_colour(sky: Colour, fog: Colour, height: Height) -> None:
    assert sky_colour(sky, fog, height) == fog


@given(colours, colours)
def test_straight_up_the_sky_is_the_sky_colour(sky: Colour, fog: Colour) -> None:
    assert sky_colour(sky, fog, STRAIGHT_UP) == sky


@given(colours, colours, heights)
def test_nowhere_is_nearer_the_sky_colour_than_straight_up(sky: Colour, fog: Colour, height: Height) -> None:
    overhead = distance(sky_colour(sky, fog, STRAIGHT_UP), sky)
    assert overhead <= distance(sky_colour(sky, fog, height), sky)


@given(heights, heights)
def test_a_higher_direction_never_has_more_fog(one: Height, other: Height) -> None:
    lower, higher = sorted((one, other))
    assert fog_share(higher) <= fog_share(lower)


@given(colours, colours, heights, heights)
def test_a_higher_direction_is_never_further_from_the_sky_colour(
    sky: Colour, fog: Colour, one: Height, other: Height
) -> None:
    lower, higher = sorted((one, other))
    above = channels(sky_colour(sky, fog, higher))
    below = channels(sky_colour(sky, fog, lower))
    for high, low, of_sky in zip(above, below, channels(sky), strict=True):
        assert abs(high - of_sky) <= abs(low - of_sky) + ROUNDING


@given(colours, colours, heights)
def test_every_channel_stays_between_the_sky_and_the_fog(sky: Colour, fog: Colour, height: Height) -> None:
    result = sky_colour(sky, fog, height)
    for channel, of_sky, of_fog in zip(channels(result), channels(sky), channels(fog), strict=True):
        assert min(of_sky, of_fog) - ROUNDING <= channel <= max(of_sky, of_fog) + ROUNDING


def test_each_channel_moves_by_the_fog_share_of_its_own_gap() -> None:
    height = rise_to_height(0.3)
    share = fog_share(height)
    result = sky_colour(NOON_SKY, NOON_FOG, height)
    for channel, of_sky, of_fog in zip(channels(result), channels(NOON_SKY), channels(NOON_FOG), strict=True):
        assert channel == pytest.approx(of_sky + (of_fog - of_sky) * share, abs=ROUNDING)


def test_the_sky_changes_just_above_the_horizon_and_just_below_straight_up() -> None:
    assert NO_FOG < fog_share(rise_to_height(0.99)) < fog_share(rise_to_height(0.01)) < ALL_FOG


def test_half_the_fog_is_left_where_the_haze_reaches() -> None:
    # Worked out by hand from share = r(1 - l) / (r + l) with r the reach squared and l the rise squared:
    # the share is a half at l = r / (1 + 2r), and (1 - r) / 2 at l = r.
    reach = HAZE_REACH * HAZE_REACH
    assert fog_share(rise_to_height(math.sqrt(reach / (1 + 2 * reach)))) == pytest.approx(0.5)
    assert fog_share(rise_to_height(HAZE_REACH)) == pytest.approx((1 - reach) / 2)


def test_most_of_the_change_is_low_in_the_sky_and_none_of_it_is_an_edge_at_the_horizon() -> None:
    assert fog_share(MOSTLY_SKY_BY) < 1 - MOST
    assert fog_share(STILL_HORIZON_AT) > NEARLY_ALL


def test_the_glsl_is_written_from_the_same_numbers_as_the_python() -> None:
    text = gradient_include().text
    assert f"PACK_SKY_HORIZON_HEIGHT = {HORIZON_HEIGHT!r};" in text
    assert f"PACK_SKY_HAZE_REACH = {HAZE_REACH!r};" in text
    assert f"vec3 {GRADIENT_FUNCTION}(vec3 skyColour, vec3 fogColour, float height)" in text


def test_the_gradient_is_an_include_that_another_effect_can_reuse() -> None:
    include = gradient_include()
    assert include.path == GRADIENT_INCLUDE_PATH
    assert effect().includes == (include,)
    assert GRADIENT_INCLUDE_LINE == f"#include <minecraft:{GRADIENT_INCLUDE_PATH.removeprefix('include/')}>"


def built_text(game: Game, path: ShaderPath) -> ShaderText:
    (file,) = (built for built in compose(game.shaders, (effect(),)) if built.path == path)
    return file.text


def test_the_effect_joins_onto_the_real_sky_shaders(game: Game) -> None:
    built = compose(game.shaders, (effect(),))
    assert [file.path for file in built] == [SKY_FRAGMENT, SKY_VERTEX, GRADIENT_INCLUDE_PATH]


def test_the_built_sky_compiles_as_the_game_compiles_it(game: Game) -> None:
    assert validate(compose(game.shaders, (effect(),)), game.shaders) == ()


def test_the_gradient_takes_the_place_of_the_fog_by_distance(game: Game) -> None:
    fragment = built_text(game, SKY_FRAGMENT)
    assert "apply_fog(" not in fragment
    assert f"{GRADIENT_FUNCTION}(ColorModulator.rgb, FogColor.rgb, packSkyHeight)" in fragment
    assert fragment.index(GRADIENT_INCLUDE_LINE) < fragment.index("void main()")


def test_the_void_below_is_mirrored_so_that_it_stays_dark(game: Game) -> None:
    assert "abs(normalize(packSkyDirection).y)" in built_text(game, SKY_FRAGMENT)


def test_both_stages_agree_on_the_direction_of_view(game: Game) -> None:
    vertex = built_text(game, SKY_VERTEX)
    assert "layout(location = 2) out vec3 packSkyDirection;" in vertex
    assert "packSkyDirection = Position;" in vertex
    assert "layout(location = 2) in vec3 packSkyDirection;" in built_text(game, SKY_FRAGMENT)
