"""Draws the game's stars as the brightest stars of the real sky, twinkling, with the lines of the constellations
and the Milky Way behind them.

How the game builds the stars (SkyRenderer.buildStars and renderStars of 26.3): it tries 1500 random
points with the fixed seed 10842 and keeps those inside the unit ball, which is 780 of them, the same
every time. Each kept star adds exactly four vertices, the corners of one quad 100 away from the camera,
to one buffer of positions. The buffer is drawn once with the shared quad index buffer, from index 0 with
base vertex 0, so gl_VertexIndex counts the vertices of that buffer and gl_VertexIndex / VERTICES_PER_STAR
is the same for the four corners of a star and for no other star.

Where a star goes. The game's positions are not used. The game's quads are all there is to draw with, so
they are shared out: the first BOX_COUNT of them are the faces of a box around the camera, on which the
Milky Way is painted, the next LINE_COUNT are the lines of the constellations' figures
(shader.effects.stars.figures), one quad stretched thin from one end of a line to the other, and the rest
are stars from the catalogue (shader.effects.stars.catalogue): first the stars that the lines join, then
the brightest of the rest (drawn_stars). The
vertex stage builds each quad itself: the number of the vertex within its quad says which corner. A line
stops short of both its ends, so that it does not run into the stars it joins. The game draws only
quads that are wound anticlockwise as the camera sees them (its own are built that way round, by JOML's
rotateTowards, and the star pipeline does not switch culling off), so the corners go round that way.

Which way the real sky lies. The game turns the star buffer about the buffer's x axis as the day goes,
and then a quarter turn about the vertical, which lays the buffer's x axis along south (renderSunMoonAndStars).
So the real sky's north pole is put on the buffer's -x axis: it lies on the northern horizon, the stars
circle it, and the sky is the one seen from the Earth's equator, where every constellation rises at some
hour. The turn from the real sky's axes to the buffer's is a pure rotation, so no constellation is mirrored.

Brightness, size and colour. A star's magnitude gives its gain and the size of its quad, evenly along the
magnitudes from the faintest row to the brightest: a magnitude is already a step the eye sees as even.
Its B-V colour index gives its tint. The tints are those of real stars pulled most of the way toward
white: a star is a few pixels, and any more colour than this reads as coloured lamps.

The stars are drawn with the game's overlay blend, which adds colour times alpha to the sky, and the game's
fade is in the colour and in the alpha alike: what reaches the eye is the fade squared. A star's gain goes
the same way, into all four channels. The fade is at most a half (the star_brightness track of the game's
day timeline), so a gain of up to 2 stays within 1 after the fade, where the game clamps a colour: the
brightest star at midnight is full white, and the game's fade still takes every star to nothing by day and
in rain.

Twinkle. The star pipeline of 26.3 is given the Globals block (RenderPipelines: STARS is built on the
snippet that holds BindGroupLayouts.GLOBALS, and SkyRenderer binds the default uniforms), which holds the
game's clock: 0 to 1 over 24000 ticks, then 0 again. A star is dimmed by two waves of the clock, each
with a whole number of cycles in one run of the clock, so nothing jumps when the clock starts again. The
cycles and where each wave starts come from a hash of the star's number, so no two stars keep time. A
twinkle only dims: the brightest star is already full white.

A dot. A star's quad is full over its middle and fades to nothing at a circle that touches its sides, so
a star is a round dot with a soft rim and not a square. A dot that fades all the way from its very centre
is mostly dim, and was lost against the Milky Way. A line fades the same way from its middle to its two
long sides, and is faint: it is there to be found, not to be seen first. A line does not twinkle.

The maths is written twice from the same constants: once here in Python, where the tests can run it, and
once as GLSL, line for line. The hash is 32 bit unsigned integer arithmetic, which Python (masked to 32
bits) and GLSL compute to the same bits. The rest is float arithmetic, 32 bit in GLSL and 64 bit here, and
differs by rounding.

The Milky Way. The star shaders are given no texture, and the sky's own shader is not told how far the
night has turned the stars, so the galaxy is drawn here, with the stars' quads. Six of them make a box
that shuts the camera in, and so there is a fragment for every pixel of the sky. The vertex stage hands
the fragment stage the place of each pixel on the box, already turned into the galaxy's axes, and
shader.effects.stars.galaxy gives the light of the galaxy in that direction. The box turns with the stars,
is added to the sky as they are, and fades with the game's fade as they do. Its corners are as far away
as the stars, since nothing is known to be drawn beyond that: a far plane nearer than a face would cut
a hole in the galaxy.

Not yet seen in the game: any of it, and above all that the quads face the camera. If no star shows, the
winding is the first thing to turn round.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import NewType

import deal

from shader.effects import Attachment, Effect, EffectName, GlslCode, Place, Position
from shader.effects.stars import catalogue, figures, galaxy
from shader.effects.stars.catalogue import CatalogueStar, ColourIndex, Magnitude
from shader.effects.stars.figures import SkyPlace
from shader.effects.stars.galaxy import GalacticDirection, Lean
from shader.vanilla import ShaderPath

# Which vertex of the star buffer, counted from 0: what gl_VertexIndex is in stars.vsh.
VertexIndex = NewType("VertexIndex", int)

# Which quad of the star buffer, counted from 0 in the order the game builds them.
StarNumber = NewType("StarNumber", int)

# Which corner of a quad, counted from 0 in the order the game draws them.
CornerNumber = NewType("CornerNumber", int)

# 32 bits mixed from the number of a star, from which its twinkle is drawn.
StarHash = NewType("StarHash", int)

# What a star multiplies the game's fade by, in colour and in alpha alike: 1 is the game's own star.
Gain = NewType("Gain", float)

# One channel of a star before the game's fade: 0 is none, 1 is the game's own star.
Level = NewType("Level", float)

# A length in the star buffer, where the game puts every star 100 from the camera.
Span = NewType("Span", float)

# The game's clock, GameTime of the Globals block: 0 to 1 over 24000 ticks, then 0 again.
Clock = NewType("Clock", float)

# A count of bits in a hash.
BitCount = NewType("BitCount", int)

# How many times a wave of a twinkle goes up and down while the clock runs once.
Cycles = NewType("Cycles", int)


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
class Direction:
    """A direction from the camera in the axes of the star buffer, or a place in it: east-west, up, north-south."""

    x: float
    y: float
    z: float


# The game adds four vertices to the star buffer for each star, the corners of one quad.
VERTICES_PER_STAR = 4

# How many quads the game draws: the points it keeps of the 1500 it tries with the seed 10842.
QUAD_COUNT = 780

# The middles of the faces of the box that the Milky Way is painted on: one each way along each axis.
BOX_FACES = (
    Direction(1.0, 0.0, 0.0),
    Direction(-1.0, 0.0, 0.0),
    Direction(0.0, 1.0, 0.0),
    Direction(0.0, -1.0, 0.0),
    Direction(0.0, 0.0, 1.0),
    Direction(0.0, 0.0, -1.0),
)

# How many of the quads are faces of the box, how many are lines, and how many are stars.
BOX_COUNT = len(BOX_FACES)
LINE_COUNT = len(figures.lines())
STAR_COUNT = QUAD_COUNT - BOX_COUNT - LINE_COUNT

# How far from the camera the game puts its stars.
STAR_DISTANCE = Span(100.0)

# A cube's corners are the square root of three times as far from its middle as its faces are. A face
# built at the stars' distance is shrunk by this, which puts the corners at that distance instead.
_BOX_SHRINK = 1.0 / math.sqrt(3.0)

# The galaxy's light is what reaches the eye at midnight, when the game's fade is a half in the colour
# and a half in the alpha. At the gain of the brightest star in both, the two halves are made up for.
GALAXY_GAIN = Gain(2.0)

# GLSL's uint is 32 bits and wraps; Python's int does not, so each product is cut to the same bits.
_UINT_BITS = BitCount(32)
_UINT_MASK = (1 << _UINT_BITS) - 1

# The hash is lowbias32 (Chris Wellons, "Prospecting for Hash Functions"): two multiplications and three
# shifts, chosen there by search for how evenly every input bit reaches every output bit. Star numbers
# are consecutive small integers, which a weaker mix turns into stars that twinkle in step.
_FIRST_SHIFT = BitCount(16)
_FIRST_MULTIPLIER = 0x7FEB352D
_SECOND_SHIFT = BitCount(15)
_SECOND_MULTIPLIER = 0x846CA68B
_THIRD_SHIFT = BitCount(16)

# The game's own star: a white one at this gain is what the game draws.
GAME_STAR = Gain(1.0)

# The faintest star of the catalogue that is drawn. It has to stand out from the faint stars of the
# Milky Way behind it, the brightest of which reach about four fifths of full white: under that, the
# stars of a constellation's figure cannot be told from the crowd.
FAINTEST = Gain(1.4)

# The brightest star: full white at midnight. More would be clamped by the game once the fade is a half.
BRIGHTEST = Gain(2.0)

# Half the side of the quad of the faintest star and of the brightest. The game's own are 0.15 to 0.25,
# drawn as full squares; a dot shows a little over half its quad at full strength.
SMALLEST = Span(0.3)
LARGEST = Span(0.8)

# How many times over a dot's fade from centre to rim is taken before it is cut off at full: at 2.5 the
# dot is full out to six tenths of the way to its rim, and fades over the rest.
DOT_FULLNESS = 2.5

# The colour index from which a star has each tint, bluest first. A star with an index under the first
# is blue-white. The steps are where the spectral classes change: A to F, G to K, K to M, and the red giants.
_TINT_FROM = (
    (ColourIndex(0.0), Tint.WHITE),
    (ColourIndex(0.6), Tint.PALE_YELLOW),
    (ColourIndex(1.0), Tint.ORANGE),
    (ColourIndex(1.5), Tint.RED),
)

_FULL = Level(1.0)

# The tints are those of real stars from hot to cool, each pulled most of the way toward white. No channel
# is above 1, so a tint never brightens.
TINT_COLOURS = {
    Tint.WHITE: StarColour(Level(1.0), Level(1.0), Level(1.0), _FULL),
    Tint.BLUE_WHITE: StarColour(Level(0.7), Level(0.82), Level(1.0), _FULL),
    Tint.PALE_YELLOW: StarColour(Level(1.0), Level(0.93), Level(0.7), _FULL),
    Tint.ORANGE: StarColour(Level(1.0), Level(0.76), Level(0.5), _FULL),
    Tint.RED: StarColour(Level(1.0), Level(0.68), Level(0.55), _FULL),
}

# The most a twinkle dims a star: by a third at the bottom of both waves.
TWINKLE_DEPTH = 0.35

# The two waves of a twinkle go round between one and two times a second, and between two and a half
# and four: 1200 seconds of clock, and a star's own count of cycles on top of the least. Two waves that
# keep no common time give a flicker that does not look like a pulse.
SLOW_CYCLES = Cycles(1200)
FAST_CYCLES = Cycles(3000)
_SLOW_BITS = BitCount(10)
_FAST_BITS = BitCount(11)
_PHASE_BITS = BitCount(6)
_PHASE_STEPS = float(1 << _PHASE_BITS)
_SLOW_PHASE_SHIFT = BitCount(_SLOW_BITS + _FAST_BITS)
_FAST_PHASE_SHIFT = BitCount(_UINT_BITS - _PHASE_BITS)

# A line of a figure: its colour, a pale blue, at a gain under the faintest star's, and half its width.
LINE_COLOUR = StarColour(Level(0.6), Level(0.75), Level(1.0), _FULL)
LINE_GAIN = Gain(0.6)
LINE_HALF_WIDTH = Span(0.1)

# How far along itself a line starts and stops: it leaves this share of its length clear at each end.
LINE_GAP = 0.12

# The corners of a quad as the game draws them, across and up from its centre. Going round them in this
# order is anticlockwise seen from the camera, which is the way round the game draws.
_CORNERS = ((1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0))

# A direction this near straight up or down has no "aside" to speak of from the vertical, so the east-west
# axis is used to find one.
_NEARLY_UPRIGHT = 0.99

# Two directions nearer than half a degree are the same star: a line's end and the star it ends at come
# from two sources that agree to a few hundredths of a degree.
_SAME_STAR = math.cos(math.radians(0.5))

_HOURS_PER_TURN = 24.0
_DEGREES_PER_TURN = 360.0

NAME = EffectName("stars")
_VERTEX_SHADER = ShaderPath("core/stars.vsh")
_FRAGMENT_SHADER = ShaderPath("core/stars.fsh")

# Each line occurs once in its shader. The varyings and the functions go before main, in both stages.
_LAST_INCLUDE = Place("#include <minecraft:projection.glsl>")
_MAIN = Place("void main() {")
_POSITION_SET = Place("gl_Position = ProjMat * ModelViewMat * vec4(Position, 1.0);")
_COLOUR_SET = Place("fragColor = ColorModulator;")

# The game's own include that declares the Globals block, which holds the clock.
GLOBALS_INCLUDE = GlslCode("#include <minecraft:globals.glsl>")


def _is_unit(direction: Direction) -> bool:
    # Multiplied and not raised to a power: a float too large to square multiplies to infinity, which is
    # simply not 1, where raising it to a power stops the program.
    return abs(math.sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z) - 1.0) <= 1e-9


def _cross(a: Direction, b: Direction) -> Direction:
    return Direction(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def _normalized(a: Direction) -> Direction:
    length = math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)
    return Direction(a.x / length, a.y / length, a.z / length)


def _combined(*scaled: tuple[Direction, float]) -> Direction:
    return Direction(
        math.fsum(d.x * s for d, s in scaled), math.fsum(d.y * s for d, s in scaled), math.fsum(d.z * s for d, s in scaled)
    )


@deal.pure
@deal.pre(lambda vertex: vertex >= 0)
@deal.ensure(lambda vertex, result: result * VERTICES_PER_STAR <= vertex < (result + 1) * VERTICES_PER_STAR)
def star_of(vertex: VertexIndex) -> StarNumber:
    """The quad that a vertex of the star buffer is a corner of."""
    return StarNumber(vertex // VERTICES_PER_STAR)


@deal.pure
@deal.pre(lambda star: 0 <= star <= _UINT_MASK)
@deal.ensure(lambda star, result: 0 <= result <= _UINT_MASK)
def star_hash(star: StarNumber) -> StarHash:
    """The bits that a star's twinkle is drawn from: the same for the same star, always."""
    mixed = int(star)
    mixed ^= mixed >> _FIRST_SHIFT
    mixed = (mixed * _FIRST_MULTIPLIER) & _UINT_MASK
    mixed ^= mixed >> _SECOND_SHIFT
    mixed = (mixed * _SECOND_MULTIPLIER) & _UINT_MASK
    mixed ^= mixed >> _THIRD_SHIFT
    return StarHash(mixed)


@deal.pure
@deal.pre(lambda place: 0 <= place.right_ascension < _HOURS_PER_TURN and -90 <= place.declination <= 90)
@deal.post(_is_unit)
def direction_of(place: SkyPlace) -> Direction:
    """The direction in the star buffer of a place in the real sky.

    The real sky's north pole goes on the buffer's -x axis, which the game lays along north, and the real
    sky turns about its pole as the buffer turns about its x axis. The other two axes follow by a pure
    rotation, so the sky is not mirrored: right ascension 0 on the equator is straight up in the buffer.
    """
    around = place.right_ascension / _HOURS_PER_TURN * math.tau
    above = place.declination / _DEGREES_PER_TURN * math.tau
    return Direction(-math.sin(above), math.cos(above) * math.cos(around), -math.cos(above) * math.sin(around))


def _dot(a: Direction, b: Direction) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def _is_on_a_figure(star: CatalogueStar, ends: tuple[Direction, ...]) -> bool:
    toward = direction_of(SkyPlace(star.right_ascension, star.declination))
    return any(_dot(toward, end) > _SAME_STAR for end in ends)


@deal.pure
@deal.post(lambda result: len(result) == STAR_COUNT and len(set(result)) == STAR_COUNT)
@deal.post(lambda result: set(result) <= set(catalogue.stars()))
def drawn_stars() -> tuple[CatalogueStar, ...]:
    """The stars that are drawn, in the order of their quads.

    There are fewer quads than the catalogue has stars, since the lines take theirs first. A line that
    ends at no star looks broken, so the stars that the figures join come first, and the quads left
    over go to the brightest of the rest.
    """
    ends = tuple(direction_of(place) for line in figures.lines() for place in (line.start, line.end))
    joined = tuple(star for star in catalogue.stars() if _is_on_a_figure(star, ends))
    others = tuple(star for star in catalogue.stars() if star not in joined)
    return (joined + others)[:STAR_COUNT]


@deal.pure
@deal.post(lambda result: 0.0 <= result <= 1.0)
def brightness(magnitude: Magnitude) -> float:
    """How bright a star is among those of the catalogue: 0 for the faintest, 1 for the brightest.

    Even along the magnitudes: a magnitude is already a step that the eye sees as even.
    """
    return min(max((catalogue.FAINTEST - magnitude) / (catalogue.FAINTEST - catalogue.BRIGHTEST), 0.0), 1.0)


@deal.pure
@deal.post(lambda result: FAINTEST <= result <= BRIGHTEST)
def gain_of(magnitude: Magnitude) -> Gain:
    """The gain of a star of a magnitude."""
    return Gain(FAINTEST + (BRIGHTEST - FAINTEST) * brightness(magnitude))


@deal.pure
@deal.post(lambda result: SMALLEST <= result <= LARGEST)
def size_of(magnitude: Magnitude) -> Span:
    """Half the side of the quad of a star of a magnitude."""
    return Span(SMALLEST + (LARGEST - SMALLEST) * brightness(magnitude))


@deal.pure
@deal.post(lambda result: result in TINT_COLOURS)
def tint_of(colour_index: ColourIndex) -> Tint:
    """The tint of a star of a colour index: blue-white for the hottest stars, red for the coolest."""
    tint = Tint.BLUE_WHITE
    for start, reached in _TINT_FROM:
        if colour_index >= start:
            tint = reached
    return tint


@deal.pure
@deal.ensure(lambda star, result: FAINTEST <= result.alpha <= BRIGHTEST)
@deal.ensure(lambda star, result: all(0.0 < level <= result.alpha for level in (result.red, result.green, result.blue)))
def colour_of(star: CatalogueStar) -> StarColour:
    """The colour of a star before the game's fade and its twinkle: its tint at its gain."""
    return TINT_COLOURS[tint_of(star.colour_index)].scaled(gain_of(star.magnitude))


@deal.pure
@deal.pre(lambda star, clock: 0 <= star <= _UINT_MASK and 0.0 <= clock <= 1.0)
@deal.post(lambda result: GAME_STAR - TWINKLE_DEPTH <= result <= GAME_STAR)
def twinkle(star: StarNumber, clock: Clock) -> Gain:
    """What a star's colour is multiplied by at a moment of the clock: 1 at its brightest, never more.

    Two waves, each with a whole number of cycles in one run of the clock, so that the star is the same
    when the clock starts again as when it ended.
    """
    mixed = star_hash(star)
    slow = SLOW_CYCLES + (mixed & ((1 << _SLOW_BITS) - 1))
    fast = FAST_CYCLES + ((mixed >> _SLOW_BITS) & ((1 << _FAST_BITS) - 1))
    slow_phase = ((mixed >> _SLOW_PHASE_SHIFT) & ((1 << _PHASE_BITS) - 1)) / _PHASE_STEPS
    fast_phase = (mixed >> _FAST_PHASE_SHIFT) / _PHASE_STEPS
    waves = math.sin(math.tau * (slow * clock + slow_phase)) + math.sin(math.tau * (fast * clock + fast_phase))
    return Gain(GAME_STAR - TWINKLE_DEPTH * (0.5 + 0.25 * waves))


@deal.pure
@deal.pre(lambda toward, half_side, corner: _is_unit(toward) and half_side > 0 and 0 <= corner < VERTICES_PER_STAR)
def star_corner(toward: Direction, half_side: Span, corner: CornerNumber) -> Direction:
    """Where a corner of a star's quad lies in the star buffer: the quad faces the camera, square on."""
    upright = Direction(0.0, 1.0, 0.0) if abs(toward.y) < _NEARLY_UPRIGHT else Direction(1.0, 0.0, 0.0)
    aside = _normalized(_cross(upright, toward))
    upward = _cross(aside, toward)
    across, up = _CORNERS[corner]
    return _combined((toward, STAR_DISTANCE), (aside, across * half_side), (upward, up * half_side))


@deal.pure
@deal.pre(lambda start, end, corner: _is_unit(start) and _is_unit(end) and 0 <= corner < VERTICES_PER_STAR)
def line_corner(start: Direction, end: Direction, corner: CornerNumber) -> Direction:
    """Where a corner of a line's quad lies in the star buffer.

    The quad runs from a little after the start of the line to a little before its end, LINE_HALF_WIDTH
    to either side. Corners 0 and 1 lie on one long side and 2 and 3 on the other.
    """
    aside = _normalized(_cross(start, end))
    across, up = _CORNERS[corner]
    # Corners 1 and 2 are the far end, and 0 and 1 the side that `aside` points to, which makes the way
    # round anticlockwise seen from the camera.
    along = LINE_GAP if up < 0 else 1.0 - LINE_GAP
    return _combined(
        (start, (1.0 - along) * STAR_DISTANCE), (end, along * STAR_DISTANCE), (aside, across * LINE_HALF_WIDTH)
    )


@deal.pure
@deal.pre(lambda face, corner: face in BOX_FACES and 0 <= corner < VERTICES_PER_STAR)
@deal.post(lambda result: abs(math.sqrt(_dot(result, result)) - STAR_DISTANCE) <= 1e-9)
def box_corner(face: Direction, corner: CornerNumber) -> Direction:
    """Where a corner of a face of the galaxy's box lies in the star buffer: as far away as the stars."""
    return _combined((star_corner(face, STAR_DISTANCE, corner), _BOX_SHRINK))


def _galactic_axes() -> tuple[Direction, Direction, Direction]:
    """The galaxy's three axes in the star buffer: toward its centre, toward its rotation, toward its pole.

    The pole is taken as given and the centre is squared up to it: the two come from the catalogue to a
    few decimal places and are not quite at a right angle.
    """
    pole = direction_of(galaxy.NORTH_POLE)
    rotation = _normalized(_cross(pole, direction_of(galaxy.CENTRE)))
    return _cross(rotation, pole), rotation, pole


@deal.pure
@deal.pre(_is_unit)
@deal.post(galaxy.is_direction)
def galactic(direction: Direction) -> GalacticDirection:
    """A direction of the star buffer in the galaxy's own axes."""
    centre, rotation, pole = _galactic_axes()
    return GalacticDirection(Lean(_dot(direction, centre)), Lean(_dot(direction, rotation)), Lean(_dot(direction, pole)))


@deal.pure
@deal.pre(lambda across, up: abs(across) <= 1.0 and abs(up) <= 1.0)
@deal.post(lambda result: 0.0 <= result <= 1.0)
def dot_strength(across: float, up: float) -> float:
    """How much of a quad's colour a place on it takes: all of it over the middle, none at the rim and beyond.

    `across` and `up` say where on the quad, from -1 at one side to 1 at the other.
    """
    fade = max(0.0, 1.0 - (across * across + up * up))
    return min(fade * fade * DOT_FULLNESS, 1.0)


def _vec3(direction: Direction) -> str:
    return f"vec3({direction.x:.6f}, {direction.y:.6f}, {direction.z:.6f})"


def _vec4(colour: StarColour) -> str:
    return f"vec4({colour.red:.4f}, {colour.green:.4f}, {colour.blue:.4f}, {colour.alpha:.4f})"


def _rows(rows: tuple[str, ...]) -> str:
    return ",\n".join(f"    {row}" for row in rows)


def _vertex_declarations() -> GlslCode:
    """The varyings, the tables of stars and lines, and the star maths of the vertex stage."""
    drawn = drawn_stars()
    places = tuple(
        f"vec4({_vec3(direction_of(SkyPlace(s.right_ascension, s.declination)))[5:-1]}, {size_of(s.magnitude):.4f})"
        for s in drawn
    )
    colours = tuple(_vec4(colour_of(s)) for s in drawn)
    starts = tuple(_vec3(direction_of(line.start)) for line in figures.lines())
    ends = tuple(_vec3(direction_of(line.end)) for line in figures.lines())
    corners = ", ".join(f"vec2({across!r}, {up!r})" for across, up in _CORNERS)
    faces = tuple(_vec3(face) for face in BOX_FACES)
    toward_centre, toward_rotation, toward_pole = _galactic_axes()
    return GlslCode(f"""\
layout(location = 0) flat out vec4 starColour;
layout(location = 1) out vec2 packFromCentre;
layout(location = 2) flat out int packIsGalaxy;
layout(location = 3) out vec3 packGalacticPlace;

const int PACK_VERTICES_PER_STAR = {VERTICES_PER_STAR};
const int PACK_BOX_COUNT = {BOX_COUNT};
const float PACK_BOX_SHRINK = {_BOX_SHRINK!r};
const vec3 PACK_GALAXY_TOWARD_CENTRE = {_vec3(toward_centre)};
const vec3 PACK_GALAXY_TOWARD_ROTATION = {_vec3(toward_rotation)};
const vec3 PACK_GALAXY_TOWARD_POLE = {_vec3(toward_pole)};
const int PACK_LINE_COUNT = {LINE_COUNT};
const int PACK_STAR_COUNT = {STAR_COUNT};
const float PACK_STAR_DISTANCE = {STAR_DISTANCE!r};
const float PACK_NEARLY_UPRIGHT = {_NEARLY_UPRIGHT!r};
const float PACK_LINE_GAP = {LINE_GAP!r};
const float PACK_LINE_HALF_WIDTH = {LINE_HALF_WIDTH!r};
const vec4 PACK_LINE_COLOUR = {_vec4(LINE_COLOUR.scaled(LINE_GAIN))};
const float PACK_GAME_STAR = {GAME_STAR!r};
const float PACK_TWINKLE_DEPTH = {TWINKLE_DEPTH!r};
const float PACK_TURN = {math.tau!r};
const vec2 PACK_CORNERS[PACK_VERTICES_PER_STAR] = vec2[PACK_VERTICES_PER_STAR]({corners});

// Outside the part of space the game draws, so a quad whose vertices are all put here has no fragment.
const vec4 PACK_NOWHERE = vec4(2.0, 2.0, 2.0, 1.0);

// Each face of the galaxy's box: the direction of its middle.
const vec3 PACK_BOX_FACES[PACK_BOX_COUNT] = vec3[PACK_BOX_COUNT](
{_rows(faces)}
);

// Each star: its direction in the star buffer, and half the side of its quad.
const vec4 PACK_STAR_PLACES[PACK_STAR_COUNT] = vec4[PACK_STAR_COUNT](
{_rows(places)}
);

// Each star's colour before the game's fade and its twinkle.
const vec4 PACK_STAR_COLOURS[PACK_STAR_COUNT] = vec4[PACK_STAR_COUNT](
{_rows(colours)}
);

// Each line of a figure: the directions of its two ends.
const vec3 PACK_LINE_STARTS[PACK_LINE_COUNT] = vec3[PACK_LINE_COUNT](
{_rows(starts)}
);
const vec3 PACK_LINE_ENDS[PACK_LINE_COUNT] = vec3[PACK_LINE_COUNT](
{_rows(ends)}
);

uint starHash(uint star) {{
    uint mixed = star;
    mixed ^= mixed >> {_FIRST_SHIFT}u;
    mixed *= {_FIRST_MULTIPLIER}u;
    mixed ^= mixed >> {_SECOND_SHIFT}u;
    mixed *= {_SECOND_MULTIPLIER}u;
    mixed ^= mixed >> {_THIRD_SHIFT}u;
    return mixed;
}}

float packTwinkle(uint star, float clock) {{
    uint mixed = starHash(star);
    float slow = float({SLOW_CYCLES}u + (mixed & {(1 << _SLOW_BITS) - 1}u));
    float fast = float({FAST_CYCLES}u + ((mixed >> {_SLOW_BITS}u) & {(1 << _FAST_BITS) - 1}u));
    float slowPhase = float((mixed >> {_SLOW_PHASE_SHIFT}u) & {(1 << _PHASE_BITS) - 1}u) / {_PHASE_STEPS!r};
    float fastPhase = float(mixed >> {_FAST_PHASE_SHIFT}u) / {_PHASE_STEPS!r};
    // Only the part of a turn matters to a sine, and taking it first keeps the float exact enough.
    float waves = sin(PACK_TURN * fract(slow * clock + slowPhase)) + sin(PACK_TURN * fract(fast * clock + fastPhase));
    return PACK_GAME_STAR - PACK_TWINKLE_DEPTH * (0.5 + 0.25 * waves);
}}

vec3 packStarCorner(vec3 toward, float halfSide, vec2 corner) {{
    vec3 upright = abs(toward.y) < PACK_NEARLY_UPRIGHT ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 aside = normalize(cross(upright, toward));
    vec3 upward = cross(aside, toward);
    return toward * PACK_STAR_DISTANCE + aside * (corner.x * halfSide) + upward * (corner.y * halfSide);
}}

vec3 packLineCorner(vec3 start, vec3 end, vec2 corner) {{
    vec3 aside = normalize(cross(start, end));
    float along = corner.y < 0.0 ? PACK_LINE_GAP : 1.0 - PACK_LINE_GAP;
    return start * ((1.0 - along) * PACK_STAR_DISTANCE) + end * (along * PACK_STAR_DISTANCE)
        + aside * (corner.x * PACK_LINE_HALF_WIDTH);
}}

""")


def _vertex_placing() -> GlslCode:
    """The quad's own position in place of the game's, and its colour, the same at all four corners.

    `packFromCentre` is what the fragment stage fades by: a star's corner as it is, so the dot is round,
    and for a line only how far to the side, so the line fades to its long sides and not toward its ends.
    A face of the galaxy's box has no colour of its own: the fragment stage works one out for each pixel
    from `packGalacticPlace`, which is the pixel's place on the box in the galaxy's axes. Turning to
    those axes is done here, at the four corners, because it is the same turn for every pixel.
    """
    return GlslCode("""\
    int packQuad = gl_VertexIndex / PACK_VERTICES_PER_STAR;
    int packLine = packQuad - PACK_BOX_COUNT;
    int packStar = packLine - PACK_LINE_COUNT;
    vec2 packCorner = PACK_CORNERS[gl_VertexIndex % PACK_VERTICES_PER_STAR];
    packIsGalaxy = 0;
    // A star and a line make no use of this, but the fragment stage takes its direction before it knows
    // which quad it has, and nowhere has no direction.
    packGalacticPlace = PACK_GALAXY_TOWARD_CENTRE;
    if (packQuad < PACK_BOX_COUNT) {
        vec3 packOnBox = packStarCorner(PACK_BOX_FACES[packQuad], PACK_STAR_DISTANCE, packCorner) * PACK_BOX_SHRINK;
        gl_Position = ProjMat * ModelViewMat * vec4(packOnBox, 1.0);
        starColour = vec4(0.0);
        packFromCentre = vec2(0.0);
        packIsGalaxy = 1;
        packGalacticPlace = vec3(dot(packOnBox, PACK_GALAXY_TOWARD_CENTRE), dot(packOnBox, PACK_GALAXY_TOWARD_ROTATION),
            dot(packOnBox, PACK_GALAXY_TOWARD_POLE));
    } else if (packLine < PACK_LINE_COUNT) {
        vec3 packStart = PACK_LINE_STARTS[packLine];
        vec3 packEnd = PACK_LINE_ENDS[packLine];
        gl_Position = ProjMat * ModelViewMat * vec4(packLineCorner(packStart, packEnd, packCorner), 1.0);
        starColour = PACK_LINE_COLOUR;
        packFromCentre = vec2(0.0, packCorner.x);
    } else if (packStar < PACK_STAR_COUNT) {
        vec4 packPlace = PACK_STAR_PLACES[packStar];
        gl_Position = ProjMat * ModelViewMat * vec4(packStarCorner(packPlace.xyz, packPlace.w, packCorner), 1.0);
        starColour = PACK_STAR_COLOURS[packStar] * packTwinkle(uint(packStar), GameTime);
        packFromCentre = packCorner;
    } else {
        gl_Position = PACK_NOWHERE;
        starColour = vec4(0.0);
        packFromCentre = vec2(0.0);
    }
""")


def _fragment_declarations() -> GlslCode:
    return GlslCode(
        f"""\
layout(location = 0) flat in vec4 starColour;
layout(location = 1) in vec2 packFromCentre;
layout(location = 2) flat in int packIsGalaxy;
layout(location = 3) in vec3 packGalacticPlace;

const float PACK_GALAXY_GAIN = {GALAXY_GAIN!r};
const float PACK_DOT_FULLNESS = {DOT_FULLNESS!r};

"""
        + galaxy.GLSL
    )


def _fragment_colouring() -> GlslCode:
    """The quad's colour under the game's fade.

    A star or a line is full over its middle and nothing at the rim (dot_strength). A face of the galaxy's
    box takes the light of the galaxy in the direction of the pixel. How much of the sky a pixel covers
    is asked for before the two part ways: the game's GLSL gives no answer to it inside a branch that
    neighbouring pixels may not take together.
    """
    return GlslCode("""\
    vec3 packToward = normalize(packGalacticPlace);
    float packPixel = max(length(dFdx(packToward)), length(dFdy(packToward)));
    if (packIsGalaxy == 1) {
        fragColor = vec4(packGalaxyLight(packToward, packPixel), 1.0) * PACK_GALAXY_GAIN * ColorModulator;
    } else {
        float packFade = max(0.0, 1.0 - dot(packFromCentre, packFromCentre));
        fragColor = starColour * ColorModulator * min(packFade * packFade * PACK_DOT_FULLNESS, 1.0);
    }
""")


@deal.pure
@deal.ensure(lambda result: result.name == NAME and not result.includes)
@deal.ensure(lambda result: {a.shader for a in result.attachments} == {_VERTEX_SHADER, _FRAGMENT_SHADER})
@deal.ensure(lambda result: sum(a.position is Position.INSTEAD for a in result.attachments) == 2)
def effect() -> Effect:
    """The stars effect: each quad placed and coloured as a face of the galaxy's box, a line of a figure or a real star."""
    return Effect(
        NAME,
        (
            Attachment(_VERTEX_SHADER, _LAST_INCLUDE, Position.AFTER, GLOBALS_INCLUDE),
            Attachment(_VERTEX_SHADER, _MAIN, Position.BEFORE, _vertex_declarations()),
            Attachment(_VERTEX_SHADER, _POSITION_SET, Position.INSTEAD, _vertex_placing()),
            Attachment(_FRAGMENT_SHADER, _MAIN, Position.BEFORE, _fragment_declarations()),
            Attachment(_FRAGMENT_SHADER, _COLOUR_SET, Position.INSTEAD, _fragment_colouring()),
        ),
    )
