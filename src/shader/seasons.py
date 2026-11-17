"""The colours a season gives grass and leaves, as the three colour maps the game tints them with.

The game looks the tint of grass and of leaves up in three textures, by the temperature and the humidity
of the biome. A season is a recipe that repaints those textures. The recipe is worked in Oklch, where a
colour is a lightness, a chroma and a hue that the eye reads evenly, so that "30 degrees toward orange"
means the same on a jungle green as on a taiga green.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import NewType

import deal

from shader.vanilla import Channel, Pixel, Texture, TexturePath

# How light a colour looks, from 0 for black to 1 for white.
Lightness = NewType("Lightness", float)

# How far a colour is from grey: 0 is grey, and the strongest colours a screen shows are near 0.3.
Chroma = NewType("Chroma", float)

# An angle on the colour wheel in degrees: red is near 30, yellow near 110, green near 140, blue near 265.
Hue = NewType("Hue", float)

# What a chroma is multiplied by: 1 leaves the colour as strong as it was.
ChromaScale = NewType("ChromaScale", float)

# How warm a biome is, from 0 for the coldest to 1 for the hottest, as the game's colour maps count it.
Temperature = NewType("Temperature", float)

# How much of a season a climate feels, from 0 for none to 1 for all of it.
Seasonality = NewType("Seasonality", float)

# One colour of light before the screen's curve is applied, from 0 to 1 when a screen can show it.
LinearLight = NewType("LinearLight", float)


class Season(Enum):
    """The time of year a pack is built for."""

    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


class MissingColourMap(Exception):
    """A colour map that a season repaints was not among the textures given."""


@dataclass(frozen=True)
class Oklch:
    """One colour as the eye reads it: how light, how strong, and which hue."""

    lightness: Lightness
    chroma: Chroma
    hue: Hue


@dataclass(frozen=True)
class Shift:
    """How a season moves a colour: the hue turned, the chroma scaled, the lightness raised.

    A negative turn takes a leaf green toward yellow, orange and red; a positive one toward blue.
    """

    hue_turn: Hue
    chroma_scale: ChromaScale
    lightness_shift: Lightness


@dataclass(frozen=True)
class Recipe:
    """A season's look: the shift of each colour map, where the climate feels the whole season."""

    grass: Shift
    foliage: Shift
    dry_foliage: Shift


GRASS_MAP = TexturePath("colormap/grass.png")
FOLIAGE_MAP = TexturePath("colormap/foliage.png")
DRY_FOLIAGE_MAP = TexturePath("colormap/dry_foliage.png")

# The three colour maps a season gives, which are the textures to ask the game for.
COLOUR_MAPS: tuple[TexturePath, ...] = (GRASS_MAP, FOLIAGE_MAP, DRY_FOLIAGE_MAP)

UNCHANGED = Shift(hue_turn=Hue(0.0), chroma_scale=ChromaScale(1.0), lightness_shift=Lightness(0.0))

# The recipes were chosen by eye, on swatches of the common biomes beside the game's own colours. Each is
# kept small enough that a forest still reads as a forest and not as a filter laid over one.
RECIPES: dict[Season, Recipe] = {
    # New leaves are thin and yellow-green, and lighter than the leaves of July.
    Season.SPRING: Recipe(
        grass=Shift(hue_turn=Hue(-5.0), chroma_scale=ChromaScale(1.05), lightness_shift=Lightness(0.03)),
        foliage=Shift(hue_turn=Hue(-7.0), chroma_scale=ChromaScale(1.0), lightness_shift=Lightness(0.05)),
        dry_foliage=Shift(hue_turn=Hue(6.0), chroma_scale=ChromaScale(0.9), lightness_shift=Lightness(0.02)),
    ),
    # The game's own colours are a summer already.
    Season.SUMMER: Recipe(grass=UNCHANGED, foliage=UNCHANGED, dry_foliage=UNCHANGED),
    # The full turn takes a leaf green to red. A warmer climate gets part of it, so its trees stop at
    # gold or orange: the variety of an autumn forest comes from the biomes and not from noise.
    Season.AUTUMN: Recipe(
        grass=Shift(hue_turn=Hue(-25.0), chroma_scale=ChromaScale(0.7), lightness_shift=Lightness(-0.03)),
        foliage=Shift(hue_turn=Hue(-90.0), chroma_scale=ChromaScale(0.78), lightness_shift=Lightness(-0.08)),
        dry_foliage=Shift(hue_turn=Hue(-14.0), chroma_scale=ChromaScale(1.2), lightness_shift=Lightness(-0.02)),
    ),
    # Frost takes the colour out before it takes the green: grey-green and pale, a little toward blue.
    Season.WINTER: Recipe(
        grass=Shift(hue_turn=Hue(8.0), chroma_scale=ChromaScale(0.3), lightness_shift=Lightness(0.05)),
        foliage=Shift(hue_turn=Hue(10.0), chroma_scale=ChromaScale(0.3), lightness_shift=Lightness(-0.01)),
        dry_foliage=Shift(hue_turn=Hue(0.0), chroma_scale=ChromaScale(0.5), lightness_shift=Lightness(0.04)),
    ),
}

# A jungle has no autumn and a birch wood has all of one. Between these two temperatures a climate feels
# a growing share of the season; the game gives a jungle 0.95 and a birch forest 0.6.
TROPICAL = Temperature(0.95)
TEMPERATE = Temperature(0.6)

NO_SEASON = Seasonality(0.0)
WHOLE_SEASON = Seasonality(1.0)

# A column of a colour map stands for the temperature at its middle, which also gives a map one pixel
# wide a temperature.
PIXEL_CENTRE = 0.5

CHANNEL_MAX = 255
BLACK = Lightness(0.0)
WHITE = Lightness(1.0)
GREY = Chroma(0.0)

# Far stronger than any colour there is, or any a shift makes of one: a real colour's chroma is under a
# half. It is a limit only so that the arithmetic has one. A chroma without limit, cubed, is too large
# for a float to hold.
STRONGEST = Chroma(1000.0)

# How far two numbers that should be the same may differ: the last bits of a sum round.
_CLOSE = 1e-9

# Larger than any angle, lightness or scale that a colour or a shift holds, by many times. It is a limit
# only so that the arithmetic has one: two numbers near the largest a float holds add up to no number.
_FARTHEST = 1e9
FULL_TURN = Hue(360.0)

# How far outside the screen's range a channel of linear light may fall and still count as shown. Turning
# a pixel into a colour and back is off by far less than this, and half a channel step is more than this
# even near black, so the slack never changes a pixel.
GAMUT_SLACK = 1e-5

# Each step halves the range of chroma searched: 16 steps find the strongest colour that shows to within
# a sixty-thousandth of the chroma asked for. 12 steps were not enough near black, where a channel step
# is a very small amount of light: the colour found stopped two channel steps short of the edge of what
# a screen shows.
GAMUT_SEARCH_STEPS = 16

# The sRGB standard's curve between a stored channel and linear light.
SRGB_LINEAR_BELOW = 0.04045
SRGB_LINEAR_SLOPE = 12.92
SRGB_OFFSET = 0.055
SRGB_GAMMA = 2.4

# Oklab, as Björn Ottosson published it: linear sRGB to the eye's three cone responses, their cube roots
# to lightness and two colour axes, and the two inverses.
ONE_THIRD = 1.0 / 3.0
Matrix = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
Triple = tuple[float, float, float]
RGB_TO_CONES: Matrix = (
    (0.4122214708, 0.5363325363, 0.0514459929),
    (0.2119034982, 0.6806995451, 0.1073969566),
    (0.0883024619, 0.2817188376, 0.6299787005),
)
CONES_TO_LAB: Matrix = (
    (0.2104542553, 0.7936177850, -0.0040720468),
    (1.9779984951, -2.4285922050, 0.4505937099),
    (0.0259040371, 0.7827717662, -0.8086757660),
)
LAB_TO_CONES: Matrix = (
    (1.0, 0.3963377774, 0.2158037573),
    (1.0, -0.1055613458, -0.0638541728),
    (1.0, -0.0894841775, -1.2914855480),
)
CONES_TO_RGB: Matrix = (
    (4.0767416621, -3.3077115913, 0.2309699292),
    (-1.2684380046, 2.6097574011, -0.3413193965),
    (-0.0041960863, -0.7034186147, 1.7076147010),
)


def _times(matrix: Matrix, vector: Triple) -> Triple:
    # Written out, not looped: this runs several times for every pixel of every map.
    top, middle, bottom = matrix
    x, y, z = vector
    return (
        top[0] * x + top[1] * y + top[2] * z,
        middle[0] * x + middle[1] * y + middle[2] * z,
        bottom[0] * x + bottom[1] * y + bottom[2] * z,
    )


def _linear(channel: Channel) -> LinearLight:
    stored = channel / CHANNEL_MAX
    if stored <= SRGB_LINEAR_BELOW:
        return LinearLight(stored / SRGB_LINEAR_SLOPE)
    return LinearLight(((stored + SRGB_OFFSET) / (1.0 + SRGB_OFFSET)) ** SRGB_GAMMA)


def _channel(light: LinearLight) -> Channel:
    # Clipped only for the slack that _shows allows, which is less than rounding takes away.
    shown = min(max(light, 0.0), 1.0)
    if shown <= SRGB_LINEAR_BELOW / SRGB_LINEAR_SLOPE:
        stored = shown * SRGB_LINEAR_SLOPE
    else:
        stored = (1.0 + SRGB_OFFSET) * shown ** (1.0 / SRGB_GAMMA) - SRGB_OFFSET
    return Channel(round(stored * CHANNEL_MAX))


def _is_shift(shift: Shift) -> bool:
    return all(abs(part) <= _FARTHEST for part in (shift.hue_turn, shift.chroma_scale, shift.lightness_shift))


def _is_colour(colour: Oklch) -> bool:
    return all(abs(part) <= _FARTHEST for part in (colour.lightness, colour.chroma, colour.hue))


@deal.pure
@deal.pre(lambda pixel: all(0 <= c <= CHANNEL_MAX for c in (pixel.red, pixel.green, pixel.blue, pixel.alpha)))
@deal.post(lambda result: BLACK <= result.lightness and result.chroma >= GREY)
@deal.post(lambda result: 0.0 <= result.hue < FULL_TURN)
@deal.ensure(lambda pixel, result: pixel_of(result, pixel.alpha) == pixel)  # nothing is lost on the way
def oklch(pixel: Pixel) -> Oklch:
    """The colour of a pixel as the eye reads it, whatever its alpha."""
    cones = _times(RGB_TO_CONES, (_linear(pixel.red), _linear(pixel.green), _linear(pixel.blue)))
    lightness, green_red, blue_yellow = _times(
        CONES_TO_LAB, (cones[0] ** ONE_THIRD, cones[1] ** ONE_THIRD, cones[2] ** ONE_THIRD)
    )
    return Oklch(
        lightness=Lightness(lightness),
        chroma=Chroma(math.hypot(green_red, blue_yellow)),
        hue=Hue(math.degrees(math.atan2(blue_yellow, green_red)) % FULL_TURN),
    )


def _light(colour: Oklch) -> Triple:
    angle = math.radians(colour.hue)
    roots = _times(
        LAB_TO_CONES, (colour.lightness, colour.chroma * math.cos(angle), colour.chroma * math.sin(angle))
    )
    return _times(CONES_TO_RGB, (roots[0] ** 3, roots[1] ** 3, roots[2] ** 3))


def _shows(colour: Oklch) -> bool:
    return all(-GAMUT_SLACK <= light <= 1.0 + GAMUT_SLACK for light in _light(colour))


def _shown(colour: Oklch) -> Oklch:
    # A shifted colour can be stronger than any screen shows: an orange cannot be as strong as the green
    # it was turned from. Cutting each channel off at its limit would bend the hue, so the colour is
    # brought toward grey instead, as little as it takes, with its hue and its lightness kept.
    if _shows(colour):
        return colour
    shows, too_strong = GREY, colour.chroma
    for _ in range(GAMUT_SEARCH_STEPS):
        between = Chroma((shows + too_strong) / 2.0)
        if _shows(Oklch(colour.lightness, between, colour.hue)):
            shows = between
        else:
            too_strong = between
    return Oklch(colour.lightness, shows, colour.hue)


@deal.pure
@deal.pre(lambda colour, alpha: 0 <= alpha <= CHANNEL_MAX)
@deal.pre(lambda colour, alpha: BLACK <= colour.lightness <= WHITE and GREY <= colour.chroma <= STRONGEST)
@deal.pre(lambda colour, alpha: math.isfinite(colour.hue))
@deal.post(lambda result: all(0 <= c <= CHANNEL_MAX for c in (result.red, result.green, result.blue)))
@deal.ensure(lambda colour, alpha, result: result.alpha == alpha)
@deal.ensure(lambda colour, alpha, result: colour.chroma != GREY or result.red == result.green == result.blue)
def pixel_of(colour: Oklch, alpha: Channel) -> Pixel:
    """The pixel of a colour with the alpha given, brought toward grey if no screen shows a colour so strong."""
    red, green, blue = _light(_shown(colour))
    return Pixel(
        red=_channel(LinearLight(red)),
        green=_channel(LinearLight(green)),
        blue=_channel(LinearLight(blue)),
        alpha=alpha,
    )


@deal.pure
@deal.post(lambda result: NO_SEASON <= result <= WHOLE_SEASON)
@deal.ensure(lambda temperature, result: (result == NO_SEASON) == (temperature >= TROPICAL))
@deal.ensure(lambda temperature, result: (result == WHOLE_SEASON) == (temperature <= TEMPERATE))
def seasonality(temperature: Temperature) -> Seasonality:
    """How much of a season a climate of this temperature feels."""
    share = (TROPICAL - temperature) / (TROPICAL - TEMPERATE)
    return Seasonality(min(max(share, NO_SEASON), WHOLE_SEASON))


@deal.pure
@deal.pre(lambda shift, share: NO_SEASON <= share <= WHOLE_SEASON and _is_shift(shift))
@deal.ensure(lambda shift, share, result: share != NO_SEASON or result == UNCHANGED)
@deal.ensure(lambda shift, share, result: math.isclose(result.hue_turn, shift.hue_turn * share))
@deal.ensure(lambda shift, share, result: math.isclose(result.lightness_shift, shift.lightness_shift * share))
# Close as numbers near the scale that changes nothing are close: a very small share of a scale added to
# that 1 and taken from it again is lost in the 1's last bits, so the two cannot be held to a share of
# their own size.
@deal.ensure(
    lambda shift, share, result: math.isclose(
        result.chroma_scale - UNCHANGED.chroma_scale,
        (shift.chroma_scale - UNCHANGED.chroma_scale) * share,
        abs_tol=_CLOSE * max(1.0, abs(shift.chroma_scale)),
    )
)
def partial(shift: Shift, share: Seasonality) -> Shift:
    """The part of a shift that a climate feeling this share of the season gets."""
    return Shift(
        hue_turn=Hue(shift.hue_turn * share),
        chroma_scale=ChromaScale(UNCHANGED.chroma_scale + (shift.chroma_scale - UNCHANGED.chroma_scale) * share),
        lightness_shift=Lightness(shift.lightness_shift * share),
    )


@deal.pure
@deal.pre(lambda colour, shift: shift.chroma_scale >= 0.0 and _is_shift(shift) and _is_colour(colour))
@deal.post(lambda result: BLACK <= result.lightness <= WHITE and 0.0 <= result.hue < FULL_TURN)
@deal.ensure(lambda colour, shift, result: math.isclose(result.chroma, colour.chroma * shift.chroma_scale))
@deal.ensure(lambda colour, shift, result: shift != UNCHANGED or not BLACK <= colour.lightness <= WHITE
             or result == colour)
def shifted(colour: Oklch, shift: Shift) -> Oklch:
    """A colour moved by a shift, kept between black and white."""
    return Oklch(
        lightness=Lightness(min(max(colour.lightness + shift.lightness_shift, BLACK), WHITE)),
        chroma=Chroma(colour.chroma * shift.chroma_scale),
        # Taken twice: an angle a hair under nothing comes out of the first as a whole turn, rounded up,
        # and a hue is always less than one.
        hue=Hue((colour.hue + shift.hue_turn) % FULL_TURN % FULL_TURN),
    )


@deal.pure
@deal.ensure(lambda texture, result: not result or len(texture.pixels) % texture.width == 0)
@deal.ensure(lambda texture, result: result or texture.width < 1 or len(texture.pixels) != texture.width * texture.height)
def is_well_formed(texture: Texture) -> bool:
    """Whether a texture has a width and as many pixels as its width and its height say."""
    return texture.width >= 1 and len(texture.pixels) == texture.width * texture.height


@deal.pure
@deal.pre(lambda texture, shift: is_well_formed(texture) and _is_shift(shift))
@deal.ensure(lambda texture, shift, result: (result.path, result.width, result.height, result.meta)
             == (texture.path, texture.width, texture.height, texture.meta))
@deal.ensure(lambda texture, shift, result: [p.alpha for p in result.pixels] == [p.alpha for p in texture.pixels])
@deal.ensure(lambda texture, shift, result: shift != UNCHANGED or result == texture)
def repainted(texture: Texture, shift: Shift) -> Texture:
    """A colour map with each column moved by the part of a shift that its temperature feels."""
    # The game reads a map at the column (1 - temperature) of the way across, so the hot biomes are on
    # the left. Humidity, the other axis, is left to the game's own colours.
    # Only the columns that hold a pixel: a map with no rows holds none however wide it says it is, and a
    # generated map of no rows and a width of millions of millions kept this busy for good.
    by_column = tuple(
        partial(shift, seasonality(Temperature(1.0 - (column + PIXEL_CENTRE) / texture.width)))
        for column in range(min(texture.width, len(texture.pixels)))
    )
    # A map is a smooth gradient, so a column holds the same pixel many times over, and every column hotter
    # than tropical or cooler than temperate has the same shift as its neighbour. Each pair of pixel and
    # shift is worked out once, which makes the game's maps about seven times quicker to repaint.
    shifts = tuple(dict.fromkeys(by_column))
    shift_of_column = tuple(shifts.index(part) for part in by_column)
    placed = tuple((pixel, shift_of_column[index % texture.width]) for index, pixel in enumerate(texture.pixels))
    painted = {
        (pixel, part): pixel_of(shifted(oklch(pixel), shifts[part]), pixel.alpha)
        for pixel, part in set(placed)
    }
    return Texture(
        path=texture.path,
        width=texture.width,
        height=texture.height,
        pixels=tuple(painted[place] for place in placed),
        meta=texture.meta,
    )


@deal.has()
@deal.pre(lambda season, vanilla: all(is_well_formed(texture) for texture in vanilla))
@deal.raises(MissingColourMap)
@deal.ensure(lambda season, vanilla, result: tuple(texture.path for texture in result) == COLOUR_MAPS)
@deal.ensure(lambda season, vanilla, result: all(
    (after.width, after.height, after.meta, tuple(p.alpha for p in after.pixels))
    == (before.width, before.height, before.meta, tuple(p.alpha for p in before.pixels))
    for after in result for before in vanilla if before.path == after.path
))
@deal.ensure(lambda season, vanilla, result: season is not Season.SUMMER or all(t in vanilla for t in result))
def colour_maps(season: Season, vanilla: tuple[Texture, ...]) -> tuple[Texture, ...]:
    """The grass, foliage and dry foliage colour maps of a season, repainted from the game's own.

    Textures that are not colour maps are passed over. MissingColourMap if one of the three is not given.
    """
    recipe = RECIPES[season]
    given = {texture.path: texture for texture in vanilla}
    shifts = {GRASS_MAP: recipe.grass, FOLIAGE_MAP: recipe.foliage, DRY_FOLIAGE_MAP: recipe.dry_foliage}
    missing = [path for path in COLOUR_MAPS if path not in given]
    if missing:
        raise MissingColourMap(f"the game's textures lack the colour map {missing[0]}")
    return tuple(repainted(given[path], shifts[path]) for path in COLOUR_MAPS)
