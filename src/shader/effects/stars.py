"""Gives each star a brightness and a colour of its own, from the number of the star.

The game draws every star the same white, ColorModulator, whose four channels are all the one fade that
day, night and rain give the stars. A real night sky is mostly faint stars with a few bright ones, and
mostly white stars with a minority that are tinted. The number of a star is all that tells one star from
another here, so both come from a hash of that number.

How the game builds the stars (SkyRenderer.buildStars and renderStars of 26.3): it tries 1500 random
points with the fixed seed 10842 and keeps those inside the unit ball, so the same stars every time. Each
kept star adds exactly four vertices, the corners of one quad, to one buffer of positions. The buffer is
drawn once with the shared quad index buffer, from index 0 with base vertex 0, so gl_VertexIndex counts
the vertices of that buffer and gl_VertexIndex / VERTICES_PER_STAR is the same for the four corners of a
star and for no other star.

A star's size is left as the game made it. To scale a quad about its centre a vertex needs the centre,
and a vertex is told only its own corner: the corner is the centre plus an offset that is turned by a
random angle the buffer does not keep, so no corner can tell where its centre is.

The stars are drawn with the game's overlay blend, which adds colour times alpha to the sky, and the fade
is in the colour and in the alpha alike: what reaches the eye is the fade squared. A star's gain goes the
same way, into all four channels, so the fragment stage stays one multiplication and what reaches the eye
is the gain squared. The fade is at most a half (the star_brightness track of the game's day timeline), so
a gain of up to 2 stays within 1 after the fade, where the game clamps a colour: the brightest star at
midnight is full white, four times the game's own, and the game's fade still takes every star to nothing
by day and in rain.

The star maths is written twice from the same constants: once here in Python, where the tests can run
it, and once as GLSL, line for line. The hash and the choice of tint are 32 bit unsigned integer
arithmetic, which Python (masked to 32 bits) and GLSL compute to the same bits. The gain is float
arithmetic: GLSL computes it in 32 bit floats and Python in 64 bit ones, so the two differ by rounding,
far below one step of a colour channel on the screen.

Not yet seen in the game: the look itself. The brightness and the tints were chosen from pictures drawn
with the game's own star positions and blend, not from the game.
"""

from dataclasses import dataclass
from enum import Enum
from typing import NewType

import deal

from shader.effects import Attachment, Effect, EffectName, GlslCode, Place, Position
from shader.vanilla import ShaderPath

# Which vertex of the star buffer, counted from 0: what gl_VertexIndex is in stars.vsh.
VertexIndex = NewType("VertexIndex", int)

# Which star, counted from 0 in the order the game builds them.
StarNumber = NewType("StarNumber", int)

# 32 bits mixed from the number of a star, from which its brightness and its tint are drawn.
StarHash = NewType("StarHash", int)

# What a star multiplies the game's fade by, in colour and in alpha alike: 1 is the game's own star.
Gain = NewType("Gain", float)

# One channel of a star before the game's fade: 0 is none, 1 is the game's own star.
Level = NewType("Level", float)

# How many stars of a hundred have one tint.
Percent = NewType("Percent", int)

# A count of bits in a hash.
BitCount = NewType("BitCount", int)


class Tint(Enum):
    """The colour family of a star, as the temperature of a real star gives it."""

    WHITE = "white"
    BLUE_WHITE = "blue-white"
    PALE_YELLOW = "pale yellow"
    ORANGE = "orange"
    RED = "red"


@dataclass(frozen=True)
class StarColour:
    """The colour of a star before the game's fade."""

    red: Level
    green: Level
    blue: Level
    alpha: Level

    def scaled(self, gain: Gain) -> "StarColour":
        """This colour at a gain: every channel times it, alpha too."""
        return StarColour(
            Level(self.red * gain), Level(self.green * gain), Level(self.blue * gain), Level(self.alpha * gain)
        )


@dataclass(frozen=True)
class _TintBand:
    """One tint: its colour at gain 1, and how many stars of a hundred have it."""

    tint: Tint
    colour: StarColour
    share: Percent


# The game adds four vertices to the star buffer for each star, the corners of its quad.
VERTICES_PER_STAR = 4

# GLSL's uint is 32 bits and wraps; Python's int does not, so each product is cut to the same bits.
_UINT_BITS = BitCount(32)
_UINT_MASK = (1 << _UINT_BITS) - 1

# The hash is lowbias32 (Chris Wellons, "Prospecting for Hash Functions"): two multiplications and three
# shifts, chosen there by search for how evenly every input bit reaches every output bit. Star numbers
# are consecutive small integers, which a weaker mix turns into visible runs of like stars.
_FIRST_SHIFT = BitCount(16)
_FIRST_MULTIPLIER = 0x7FEB352D
_SECOND_SHIFT = BitCount(15)
_SECOND_MULTIPLIER = 0x846CA68B
_THIRD_SHIFT = BitCount(16)

# The gain is drawn from the low half of the hash and the tint from the high half, so that neither says
# anything about the other. 16 bits is far finer than the eye or the screen can tell apart, and a 16 bit
# integer divided by a power of two is exact in a 32 bit float.
DRAW_BITS = BitCount(16)
DRAW_STEPS = 1 << DRAW_BITS
_DRAW_MASK = DRAW_STEPS - 1

# The game's own star: a white one at this gain is what the game draws.
GAME_STAR = Gain(1.0)

# The faintest star, a little under the game's own. Much fainter and a star is lost against the night sky,
# which leaves the sky emptier, not deeper.
FAINTEST = Gain(0.8)

# The brightest star: full white at midnight. More would be clamped by the game once the fade is a half.
BRIGHTEST = Gain(2.0)

# The gain rises from FAINTEST to BRIGHTEST with this power of a uniform draw. The higher the power, the
# more stars stay near FAINTEST. With 4, more than half the stars are within a tenth of the way up and one
# in six is past halfway: mostly faint stars with a few bright ones, which is the shape of a real sky.
STEEPNESS = 4.0

# A tint leaves alpha alone, so that a star's gain alone says how much of the star is added to the sky.
_FULL = Level(1.0)

_ALL = Percent(100)

# White first: the band a star falls in is counted from the bands whose first draw it has reached, and
# the order here is the order of the colours in the GLSL array. The tints are those of real stars from
# hot to cool, each pulled most of the way toward white: a star is a few pixels, and any more colour
# than this reads as coloured lamps, not as stars. No channel is above 1, so a tint never brightens.
_BANDS = (
    _TintBand(Tint.WHITE, StarColour(Level(1.0), Level(1.0), Level(1.0), _FULL), Percent(62)),
    _TintBand(Tint.BLUE_WHITE, StarColour(Level(0.7), Level(0.82), Level(1.0), _FULL), Percent(17)),
    _TintBand(Tint.PALE_YELLOW, StarColour(Level(1.0), Level(0.93), Level(0.7), _FULL), Percent(12)),
    _TintBand(Tint.ORANGE, StarColour(Level(1.0), Level(0.76), Level(0.5), _FULL), Percent(6)),
    _TintBand(Tint.RED, StarColour(Level(1.0), Level(0.68), Level(0.55), _FULL), Percent(3)),
)

# The colour of each tint at gain 1.
TINT_COLOURS = {band.tint: band.colour for band in _BANDS}

# How many stars of a hundred have each tint.
TINT_SHARES = {band.tint: band.share for band in _BANDS}


@deal.pure
@deal.pre(lambda before: 0 <= before <= len(_BANDS))
@deal.ensure(lambda before, result: 0 <= result <= DRAW_STEPS)
def _first_draw(before: int) -> int:
    """The first tint draw past the shares of that many bands, counted from the first band."""
    return sum(band.share for band in _BANDS[:before]) * DRAW_STEPS // _ALL


# The first draw of each band after the first; a draw has reached a band when it is at least this.
_BAND_STARTS = tuple(_first_draw(before) for before in range(1, len(_BANDS)))

_NAME = EffectName("stars")
_VERTEX_SHADER = ShaderPath("core/stars.vsh")
_FRAGMENT_SHADER = ShaderPath("core/stars.fsh")

# Each line occurs once in its shader. The varying and the functions go before main, in both stages.
_MAIN = Place("void main() {")
_POSITION_SET = Place("gl_Position = ProjMat * ModelViewMat * vec4(Position, 1.0);")
_COLOUR_SET = Place("fragColor = ColorModulator;")

# stars.vsh has no out and stars.fsh no in of its own, so location 0 is free between the two stages.
_VARYING_LOCATION = 0


@deal.pure
@deal.pre(lambda vertex: vertex >= 0)
@deal.ensure(lambda vertex, result: result * VERTICES_PER_STAR <= vertex < (result + 1) * VERTICES_PER_STAR)
def star_of(vertex: VertexIndex) -> StarNumber:
    """The star that a vertex of the star buffer is a corner of."""
    return StarNumber(vertex // VERTICES_PER_STAR)


@deal.pure
@deal.pre(lambda star: 0 <= star <= _UINT_MASK)
@deal.ensure(lambda star, result: 0 <= result <= _UINT_MASK)
def star_hash(star: StarNumber) -> StarHash:
    """The bits that a star's brightness and tint are drawn from: the same for the same star, always."""
    mixed = int(star)
    mixed ^= mixed >> _FIRST_SHIFT
    mixed = (mixed * _FIRST_MULTIPLIER) & _UINT_MASK
    mixed ^= mixed >> _SECOND_SHIFT
    mixed = (mixed * _SECOND_MULTIPLIER) & _UINT_MASK
    mixed ^= mixed >> _THIRD_SHIFT
    return StarHash(mixed)


@deal.pure
@deal.pre(lambda mixed: 0 <= mixed <= _UINT_MASK)
@deal.ensure(lambda mixed, result: FAINTEST <= result < BRIGHTEST)
def gain_of(mixed: StarHash) -> Gain:
    """The gain of the star of a hash: near FAINTEST for most stars, toward BRIGHTEST for a few."""
    draw = (mixed & _DRAW_MASK) / DRAW_STEPS
    return Gain(FAINTEST + (BRIGHTEST - FAINTEST) * draw**STEEPNESS)


@deal.pure
@deal.pre(lambda mixed: 0 <= mixed <= _UINT_MASK)
@deal.ensure(lambda mixed, result: result in TINT_COLOURS)
def tint_of(mixed: StarHash) -> Tint:
    """The tint of the star of a hash: white for most stars."""
    draw = mixed >> DRAW_BITS
    band = sum(draw >= start for start in _BAND_STARTS)
    return _BANDS[band].tint


@deal.pure
@deal.pre(lambda star: 0 <= star <= _UINT_MASK)
@deal.ensure(lambda star, result: FAINTEST <= result.alpha < BRIGHTEST)
@deal.ensure(lambda star, result: all(0.0 < level <= result.alpha for level in (result.red, result.green, result.blue)))
def colour_of(star: StarNumber) -> StarColour:
    """The colour of a star before the game's fade: its tint at its gain."""
    mixed = star_hash(star)
    return TINT_COLOURS[tint_of(mixed)].scaled(gain_of(mixed))


@deal.pure
def _vec4(colour: StarColour) -> str:
    return f"vec4({colour.red!r}, {colour.green!r}, {colour.blue!r}, {colour.alpha!r})"


@deal.pure
def _uint(value: int) -> str:
    return f"{value}u"


@deal.pure
def _vertex_declarations() -> GlslCode:
    """The varying and the star maths of the vertex stage, mirroring the Python above line for line."""
    tints = ", ".join(_vec4(band.colour) for band in _BANDS)
    reached = " + ".join(f"int(draw >= {_uint(start)})" for start in _BAND_STARTS)
    return GlslCode(f"""\
layout(location = {_VARYING_LOCATION}) flat out vec4 starColour;

const vec4 STAR_TINTS[{len(_BANDS)}] = vec4[{len(_BANDS)}]({tints});

uint starHash(uint star) {{
    uint mixed = star;
    mixed ^= mixed >> {_uint(_FIRST_SHIFT)};
    mixed *= {_uint(_FIRST_MULTIPLIER)};
    mixed ^= mixed >> {_uint(_SECOND_SHIFT)};
    mixed *= {_uint(_SECOND_MULTIPLIER)};
    mixed ^= mixed >> {_uint(_THIRD_SHIFT)};
    return mixed;
}}

float starGain(uint mixed) {{
    float draw = float(mixed & {_uint(_DRAW_MASK)}) / {float(DRAW_STEPS)!r};
    return {FAINTEST!r} + {BRIGHTEST - FAINTEST!r} * pow(draw, {STEEPNESS!r});
}}

vec4 starTint(uint mixed) {{
    uint draw = mixed >> {_uint(DRAW_BITS)};
    int band = {reached};
    return STAR_TINTS[band];
}}

""")


@deal.pure
def _vertex_colouring() -> GlslCode:
    """The colour of the vertex's star, the same at all four corners so that flat reads it whole."""
    return GlslCode(f"""\
    uint starMixed = starHash(uint(gl_VertexIndex / {VERTICES_PER_STAR}));
    starColour = starTint(starMixed) * starGain(starMixed);
""")


@deal.pure
def _fragment_declarations() -> GlslCode:
    return GlslCode(f"layout(location = {_VARYING_LOCATION}) flat in vec4 starColour;\n\n")


@deal.pure
def _fragment_colouring() -> GlslCode:
    """The star's colour under the game's fade: one multiplication, which is all a star's pixel costs."""
    return GlslCode("    fragColor = starColour * ColorModulator;\n")


@deal.pure
@deal.ensure(lambda result: result.name == _NAME and not result.includes)
@deal.ensure(lambda result: {a.shader for a in result.attachments} == {_VERTEX_SHADER, _FRAGMENT_SHADER})
@deal.ensure(lambda result: sum(a.position is Position.INSTEAD for a in result.attachments) == 1)
def effect() -> Effect:
    """The stars effect: each star's colour worked out in stars.vsh and drawn, under the game's fade, by stars.fsh."""
    return Effect(
        _NAME,
        (
            Attachment(_VERTEX_SHADER, _MAIN, Position.BEFORE, _vertex_declarations()),
            Attachment(_VERTEX_SHADER, _POSITION_SET, Position.AFTER, _vertex_colouring()),
            Attachment(_FRAGMENT_SHADER, _MAIN, Position.BEFORE, _fragment_declarations()),
            Attachment(_FRAGMENT_SHADER, _COLOUR_SET, Position.INSTEAD, _fragment_colouring()),
        ),
    )
