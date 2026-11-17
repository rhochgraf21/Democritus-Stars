"""The Milky Way: the colour of its light in every direction of the real sky.

The game gives the star shaders no texture, so the galaxy is not a photograph but a model, worked out for
each pixel. What is real in it is where things are: the band lies along the real galactic equator, the
bulge is at the real centre of the galaxy in Sagittarius, the band is brightest toward the centre and
faintest opposite it, and the two Magellanic Clouds are where they are in the southern sky. What is made
up is the detail: the star clouds, the dust, the nebulae and the faint stars come from noise, placed
where the real ones gather (dust along the middle of the band and mostly on the centre's side, nebulae
in the plane, faint stars everywhere and crowded where the band is bright).

The light is of two kinds. The haze is the light of stars too many and too faint to be told apart, and
it is smooth. The faint stars are points, each a pixel or two across: a photograph of the Milky Way is
mostly these, and without them the band is a smear. They are not the stars of the catalogue, which the
stars effect draws on top with quads of their own.

The colours are those of a long photograph and not of the eye, which sees the Milky Way grey: gold for the
old stars of the bulge, blue-white for the young stars of the disc, brown where dust takes the blue out
of the light behind it, and pink for the glowing hydrogen of the nebulae.

A direction is kept in the galaxy's own axes: toward the centre, toward where the galaxy turns, and toward
its north pole. The noise is a lattice in those three axes and not in the two angles of a map of the sky,
so it has no seam where the longitude starts again and does not bunch up at the poles.

The maths lives twice: as the functions here, which the tests check, and as GLSL, which the game runs.
Each GLSL function mirrors the Python one line for line. Change one and change the other. The hashes are
32 bit integer arithmetic and agree to the bit; the rest is float arithmetic and differs by rounding.

What a pixel costs is the noise, so it is kept cheap: a corner of a lattice is one mix of its three
numbers and not a mix for each, and one bend of the sky serves the dust, the mottle, the wander of the
dust lane and the colour of the disc alike.

The brightness and the colours were chosen from pictures drawn by the Python functions, on a sky as dark
as the game's at midnight.
"""

import math
from dataclasses import dataclass
from typing import NewType

import deal

from shader.effects import GlslCode
from shader.effects.stars.catalogue import Degrees, Hours
from shader.effects.stars.figures import SkyPlace

# The light of one colour that reaches the eye at midnight: 0 is none, 1 is all a screen can show.
Light = NewType("Light", float)

# A value of noise, or a point on its range: 0 to 1.
Level = NewType("Level", float)

# How much of something: 0 is none of it, 1 is all of it.
Share = NewType("Share", float)

# 32 bits, as a hash takes and gives them.
Bits = NewType("Bits", int)

# What tells one noise from another over the same lattice.
Seed = NewType("Seed", int)

# How many cells of a lattice lie along the radius of the sky's sphere. A cell is 57 / frequency degrees.
Frequency = NewType("Frequency", float)

# How many lattices are laid over one another, each twice as fine and half as strong as the one before.
Octaves = NewType("Octaves", int)

# How far a direction points along one axis: -1 is straight against it, 1 straight along it.
Lean = NewType("Lean", float)

# A length on the sky's sphere, whose radius is 1. Near the equator it is the sine of the latitude.
Reach = NewType("Reach", float)


@dataclass(frozen=True)
class GalacticDirection:
    """A direction from the Sun in the galaxy's axes, a vector of length 1, or a place a little off the sky's sphere."""

    toward_centre: Lean
    toward_rotation: Lean
    toward_pole: Lean


@dataclass(frozen=True)
class Glow:
    """The light of the galaxy in one direction, in red, green and blue."""

    red: Light
    green: Light
    blue: Light

    def scaled(self, share: float) -> "Glow":
        """This glow with every colour times a number."""
        return Glow(Light(self.red * share), Light(self.green * share), Light(self.blue * share))

    def plus(self, other: "Glow") -> "Glow":
        """This glow and another shining together."""
        return Glow(Light(self.red + other.red), Light(self.green + other.green), Light(self.blue + other.blue))

    def blended(self, other: "Glow", share: Share) -> "Glow":
        """This glow turned toward another by a share: none of the other at 0, all of it at 1."""
        return self.scaled(1.0 - share).plus(other.scaled(share))

    def dimmed(self, taken: "Glow") -> "Glow":
        """This glow after a share of each colour is taken out of it."""
        return Glow(
            Light(self.red * (1.0 - taken.red)), Light(self.green * (1.0 - taken.green)), Light(self.blue * (1.0 - taken.blue))
        )

    def capped(self) -> "Glow":
        """This glow with no colour above what a screen can show."""
        return Glow(Light(min(self.red, FULL)), Light(min(self.green, FULL)), Light(min(self.blue, FULL)))


@dataclass(frozen=True)
class Satellite:
    """A small galaxy beside the Milky Way: where its middle is, and how far from it its light reaches."""

    toward: GalacticDirection
    reach: Reach


@dataclass(frozen=True)
class StarSheet:
    """One sheet of faint stars: how fine its lattice is, how bright its brightest star, and its seed."""

    frequency: Frequency
    light: Light
    seed: Seed


@dataclass(frozen=True)
class Bend:
    """How far the sky is bent at a place, along each of the galaxy's axes: each from -0.5 to 0.5."""

    toward_centre: float
    toward_rotation: float
    toward_pole: float


# The centre of the galaxy and its north pole in the real sky, for the year 2000 (IAU 1958 galactic
# coordinates, as Reid and Brunthaler 2004 give them for J2000).
CENTRE = SkyPlace(Hours(17.76033), Degrees(-28.93617))
NORTH_POLE = SkyPlace(Hours(12.85730), Degrees(27.12834))

FULL = Light(1.0)
DARK = Glow(Light(0.0), Light(0.0), Light(0.0))

_DEGREES_PER_TURN = 360.0


def _toward(longitude: Degrees, latitude: Degrees) -> GalacticDirection:
    around = longitude / _DEGREES_PER_TURN * math.tau
    above = latitude / _DEGREES_PER_TURN * math.tau
    return GalacticDirection(
        Lean(math.cos(above) * math.cos(around)), Lean(math.cos(above) * math.sin(around)), Lean(math.sin(above))
    )


# The Large and the Small Magellanic Cloud, at their galactic longitude and latitude. Their reaches are
# about 4 and 2 degrees: what a dark sky shows of them.
SATELLITES = (
    Satellite(_toward(Degrees(280.47), Degrees(-32.89)), Reach(0.075)),
    Satellite(_toward(Degrees(302.80), Degrees(-44.30)), Reach(0.04)),
)

# The band opposite the centre, as a share of the band toward the centre. The Sun is well out from the
# middle of the disc, so there are fewer stars to look through on the way out than on the way in.
ANTICENTRE = Share(0.35)

# The band is a thin bright layer inside a wide faint one: their half heights, about 6 and 17 degrees,
# and the share of the light at the equator that the thin one gives.
THIN_HEIGHT = Reach(0.11)
WIDE_HEIGHT = Reach(0.3)
THIN_SHARE = Share(0.6)

# The bulge about the centre: how far it reaches along the band and above it, about 21 and 12 degrees.
BULGE_LENGTH = Reach(0.36)
BULGE_HEIGHT = Reach(0.21)

# Where the band gives less than this, no haze is drawn and no noise is worked out, only the faint stars:
# most of the sky is far from the band, and the noise is what a pixel costs.
FAINTEST_DRAWN = Light(0.004)

# The bend: three noises, one for each axis, by which a place is moved before the dust is looked
# up. Noise looked up at bent places runs in streams and curls, as gas does; looked up
# straight it makes round blobs. BEND_REACH is how far a place is moved at most, each way.
BEND_FREQUENCY = Frequency(4.0)
BEND_REACH = Reach(0.1)
BEND_SEEDS = (Seed(31), Seed(37), Seed(41))

# The star clouds: noise that the band is multiplied by, from a floor to a floor plus a range, so that
# the average stays about 1. Its coarsest cells are about 19 degrees, the size of the great star clouds.
# Noise of several octaves keeps near its middle, so the levels between CLUMPS_FROM and CLUMPS_FULL are
# spread over the whole range: without that the band is even, and a star cloud cannot be told from a gap.
CLUMP_FREQUENCY = Frequency(3.0)
CLUMP_OCTAVES = Octaves(4)
CLUMPS_FROM = Level(0.3)
CLUMPS_FULL = Level(0.7)
CLUMP_FLOOR = 0.3
CLUMP_RANGE = 1.4
CLUMP_SEED = Seed(11)

# The mottle: fine noise, about a degree and a half across, that the haze is multiplied by on top of the
# star clouds, from a floor to a floor plus a range. Without it the haze between the dust is as smooth
# as fog, and it hides the steps of an 8 bit screen in the faint light. It is looked up at the bent place:
# looked up straight, the squares of so fine a lattice show as a grid in the faint parts of the band.
MOTTLE_FREQUENCY = Frequency(38.0)
MOTTLE_FLOOR = 0.72
MOTTLE_RANGE = 0.56
MOTTLE_SEED = Seed(17)

# The dust: a dark lane along the band whose middle wanders to either side of the equator with the bend.
# How thick the dust is at a place is the lane times ridged noise, which is high along thin winding
# lines, and the dust is there where that is high: so the lane is a net of filaments with ragged edges
# and gaps, as the Great Rift is, and not a stripe painted on the band. DUST_DEPTH is the share of the
# light that the thickest dust takes, and DUST_ANTICENTRE how much of the dust is left opposite the
# centre, where there is less of the disc to look through.
DUST_WANDER = Reach(0.3)
DUST_HEIGHT = Reach(0.13)
DUST_FREQUENCY = Frequency(5.0)
DUST_OCTAVES = Octaves(5)
DUST_FROM = Level(0.3)
DUST_FULL = Level(0.64)
DUST_DEPTH = Share(0.9)
DUST_ANTICENTRE = Share(0.4)
DUST_SEED = Seed(43)

# The share of each colour that dust takes of the light behind it, at its thickest: all of the blue and
# less of the red, which is why a dust lane is brown at its edges and not grey.
DUST_TAKES = Glow(Light(0.78), Light(0.9), Light(1.0))

# The nebulae, kept to the plane of the galaxy. Where fine noise is at its highest hydrogen glows: red at
# the rim of a nebula and pink-white in its middle. Where the same noise is at its lowest a cloud of
# dust shines blue with the light of the stars beside it.
NEBULA_FREQUENCY = Frequency(24.0)
NEBULA_OCTAVES = Octaves(2)
NEBULA_FROM = Level(0.66)
NEBULA_FULL = Level(0.9)
NEBULA_HEIGHT = Reach(0.06)
REFLECTION_FROM = Level(0.3)
REFLECTION_FULL = Level(0.16)
NEBULA_SEED = Seed(53)
NEBULA_RIM = Glow(Light(0.42), Light(0.05), Light(0.13))
NEBULA_MIDDLE = Glow(Light(0.85), Light(0.42), Light(0.55))
REFLECTION_COLOUR = Glow(Light(0.06), Light(0.14), Light(0.3))

# The haze of each part where it is brightest, before the star clouds: the disc toward the centre on the
# equator, which the bend turns between a blue and a violet from place to place, the bulge, the cream of
# its very middle on top of that, and the middle of a satellite.
DISC_BLUE = Glow(Light(0.17), Light(0.27), Light(0.5))
DISC_VIOLET = Glow(Light(0.3), Light(0.26), Light(0.44))
BULGE_COLOUR = Glow(Light(0.5), Light(0.34), Light(0.16))
CORE_COLOUR = Glow(Light(0.26), Light(0.24), Light(0.18))
SATELLITE_COLOUR = Glow(Light(0.24), Light(0.3), Light(0.44))

# The faint stars: two sheets, the second finer and fainter. A sheet is a lattice with one star to a
# cell, at most. A cell is 57 / frequency degrees, so the sheets hold some tens of thousands of stars
# each over the whole sky, of which the crowding lets a share be seen.
STAR_SHEETS = (
    StarSheet(Frequency(110.0), Light(0.8), Seed(71)),
    StarSheet(Frequency(190.0), Light(0.5), Seed(73)),
)

# The share of the cells that hold a star: away from the band, for each unit of the band's light on top
# of that, and at most. More than the most and the stars are a grid, since there is one to a cell.
CROWD_EVERYWHERE = Share(0.08)
CROWD_IN_BAND = Share(1.6)
CROWD_MOST = Share(0.85)

# A star's middle keeps this far from the walls of its cell, as a share of the cell, and a star is never
# wider than that: so no star is cut off at a wall, and a pixel need look only into the cell it is in.
STAR_MARGIN = 0.3

# How far a faint star's light reaches from its middle: this many pixels, and never less than this much
# of the sky. A star narrower than a pixel flickers as the sky turns.
STAR_PIXELS = 1.3
STAR_REACH = Reach(0.0014)

# A faint star's light is from this share of its sheet's brightest to all of it, by the fourth power of
# an even chance: most stars are faint and a few are bright, as in the sky.
STAR_FAINTEST = Share(0.1)

# The two colours a faint star lies between: a hot star and a cool one.
STAR_HOT = Glow(Light(0.72), Light(0.84), Light(1.0))
STAR_COOL = Glow(Light(1.0), Light(0.84), Light(0.62))

# A direction has parts from -1 to 1, and a bent place lies within BEND_REACH of one. Moved up by this,
# every part is above 0, so the corner of the lattice that a place belongs to is found by rounding
# down, and its number is never negative.
_LATTICE_OFFSET = 2.0
_LATTICE_REACH = 1.5

# The finest lattice and the most octaves that noise is made of. At both together a corner's number is
# still well inside what a 32 bit float holds exactly, and a cell is far smaller than a pixel of any
# screen, so nothing finer could be seen.
FINEST = Frequency(4096.0)
MOST_OCTAVES = Octaves(8)

# The two numbers of the smoothstep curve t * t * (3 - 2 * t), which is level at both ends: blended with
# it, noise has no crease along the lines of its lattice.
_SMOOTH_OFFSET = 3.0
_SMOOTH_SLOPE = 2.0

# The 32 bit mixer "lowbias32" of Chris Wellons' hash prospector: three shifts and two multipliers found
# by search to make every input bit change every output bit about half the time.
_WORD = Bits(0xFFFFFFFF)
_MIX_SHIFTS = (16, 15, 16)
_MIX_MULTIPLIERS = (Bits(0x7FEB352D), Bits(0x846CA68B))

# A corner's three numbers and its seed are each multiplied by a large odd number of its own (the primes
# of xxHash32) and folded together before one mix. A mix for each number would be four times the work
# for every corner of every lattice of every pixel, and the picture does not show the difference.
_FOLD_MULTIPLIERS = (Bits(0x9E3779B1), Bits(0x85EBCA77), Bits(0xC2B2AE3D), Bits(0x27D4EB2F))

# A level is the top 24 bits of a hash, because a 32 bit float holds 24 bits exactly.
_LEVEL_BITS = 24
_DROPPED_BITS = 32 - _LEVEL_BITS
_LEVEL_STEPS = float(1 << _LEVEL_BITS)

# A cell's hash is cut into four bytes: where the star's middle is along each axis, and its chance.
_BYTE_BITS = 8
_BYTE_MASK = Bits(0xFF)
_BYTE_STEPS = 256.0
_CHANCE_SHIFT = 3 * _BYTE_BITS

# A second hash of the cell is cut into two halves: how bright the star is, and how cool.
_HALF_BITS = 16
_HALF_MASK = Bits(0xFFFF)
_HALF_STEPS = 65536.0

# A cell of a lattice in three axes has eight corners: each bit of a corner's number says whether the
# corner is the near or the far one along one axis.
_CELL_CORNERS = 8

_UNIT_SLACK = 1e-6


def _parts(place: GalacticDirection) -> tuple[float, float, float]:
    return (place.toward_centre, place.toward_rotation, place.toward_pole)


def _length(direction: GalacticDirection) -> float:
    return math.sqrt(math.fsum(part * part for part in _parts(direction)))


@deal.pure
@deal.ensure(lambda direction, result: result == (abs(_length(direction) - 1.0) <= _UNIT_SLACK))
def is_direction(direction: GalacticDirection) -> bool:
    """Whether a direction has the length 1, which the light of the galaxy is worked out for."""
    return abs(_length(direction) - 1.0) <= _UNIT_SLACK


@deal.pure
@deal.ensure(lambda place, result: not result or max(abs(part) for part in _parts(place)) <= _LATTICE_REACH)
@deal.ensure(lambda place, result: result or any(not abs(part) <= _LATTICE_REACH for part in _parts(place)))
def is_on_the_lattice(place: GalacticDirection) -> bool:
    """Whether a place is near enough to the sky's sphere for noise to be looked up at it.

    A place with a part that is not a number is not: no comparison with it holds.
    """
    return all(abs(part) <= _LATTICE_REACH for part in _parts(place))


@deal.pure
@deal.ensure(lambda frequency, octaves, result: result == (0 < frequency <= FINEST and 0 < octaves <= MOST_OCTAVES))
def is_lattice(frequency: Frequency, octaves: Octaves) -> bool:
    """Whether noise can be made of lattices from this frequency on, this many octaves of them."""
    return 0 < frequency <= FINEST and 0 < octaves <= MOST_OCTAVES


def _is_glow(glow: Glow) -> bool:
    return all(0.0 <= light <= FULL for light in (glow.red, glow.green, glow.blue))


def _mixed(bits: Bits) -> Bits:
    bits = Bits(bits ^ (bits >> _MIX_SHIFTS[0]))
    bits = Bits(bits * _MIX_MULTIPLIERS[0] & _WORD)
    bits = Bits(bits ^ (bits >> _MIX_SHIFTS[1]))
    bits = Bits(bits * _MIX_MULTIPLIERS[1] & _WORD)
    return Bits(bits ^ (bits >> _MIX_SHIFTS[2]))


def _corner_bits(corner: tuple[int, int, int], seed: Seed) -> Bits:
    folded = (
        corner[0] * _FOLD_MULTIPLIERS[0]
        ^ corner[1] * _FOLD_MULTIPLIERS[1]
        ^ corner[2] * _FOLD_MULTIPLIERS[2]
        ^ seed * _FOLD_MULTIPLIERS[3]
    )
    return _mixed(Bits(folded & _WORD))


def _corner_level(corner: tuple[int, int, int], seed: Seed) -> Level:
    return Level((_corner_bits(corner, seed) >> _DROPPED_BITS) / _LEVEL_STEPS)


def _bell(distance: float, reach: Reach) -> Share:
    """1 at no distance, falling away smoothly: about a third at the reach, and next to nothing at twice it."""
    return Share(math.exp(-(distance / reach) * (distance / reach)))


def _eased(low: Level, high: Level, value: Level) -> Share:
    """0 at `low` and 1 at `high`, level at both. `low` may be the higher of the two, which turns it round."""
    toward = min(max((value - low) / (high - low), 0.0), 1.0)
    return Share(toward * toward * (_SMOOTH_OFFSET - _SMOOTH_SLOPE * toward))


@deal.pure
@deal.pre(lambda place, frequency, seed: is_on_the_lattice(place) and is_lattice(frequency, Octaves(1)))
@deal.post(lambda result: 0.0 <= result <= 1.0)
def smooth_noise(place: GalacticDirection, frequency: Frequency, seed: Seed) -> Level:
    """Noise over the sky: a level for every place, the same each time, with no seam and no crease."""
    steps = tuple((part + _LATTICE_OFFSET) * frequency for part in _parts(place))
    whole = tuple(math.floor(step) for step in steps)
    toward = tuple(step - floor for step, floor in zip(steps, whole))
    blend = tuple(t * t * (_SMOOTH_OFFSET - _SMOOTH_SLOPE * t) for t in toward)
    level = 0.0
    for corner in range(_CELL_CORNERS):
        far = (corner & 1, (corner >> 1) & 1, (corner >> 2) & 1)
        weight = math.prod(b if f else 1.0 - b for b, f in zip(blend, far))
        level += weight * _corner_level((whole[0] + far[0], whole[1] + far[1], whole[2] + far[2]), seed)
    return Level(min(level, 1.0))


@deal.pure
@deal.pre(lambda place, frequency, octaves, seed: is_on_the_lattice(place) and is_lattice(frequency, octaves))
@deal.post(lambda result: 0.0 <= result <= 1.0)
def clouded_noise(place: GalacticDirection, frequency: Frequency, octaves: Octaves, seed: Seed) -> Level:
    """Noise with detail at several sizes, as a cloud has: lattices over one another, each twice as fine and half as strong."""
    level = 0.0
    strength = 1.0
    total = 0.0
    for octave in range(octaves):
        level += strength * smooth_noise(place, Frequency(frequency * (1 << octave)), Seed(seed + octave))
        total += strength
        strength *= 0.5
    return Level(min(level / total, 1.0))


@deal.pure
@deal.pre(lambda place, frequency, octaves, seed: is_on_the_lattice(place) and is_lattice(frequency, octaves))
@deal.post(lambda result: 0.0 <= result <= 1.0)
def ridged_noise(place: GalacticDirection, frequency: Frequency, octaves: Octaves, seed: Seed) -> Level:
    """Noise that is high along thin winding lines, as filaments of dust are, with detail at several sizes.

    Each lattice is folded about its middle level, which is where it is highest, and squared, which
    narrows the lines.
    """
    level = 0.0
    strength = 1.0
    total = 0.0
    for octave in range(octaves):
        ridge = 1.0 - abs(2.0 * smooth_noise(place, Frequency(frequency * (1 << octave)), Seed(seed + octave)) - 1.0)
        level += strength * ridge * ridge
        total += strength
        strength *= 0.5
    return Level(min(level / total, 1.0))


def _bend(direction: GalacticDirection) -> Bend:
    return Bend(
        smooth_noise(direction, BEND_FREQUENCY, BEND_SEEDS[0]) - 0.5,
        smooth_noise(direction, BEND_FREQUENCY, BEND_SEEDS[1]) - 0.5,
        smooth_noise(direction, BEND_FREQUENCY, BEND_SEEDS[2]) - 0.5,
    )


def _bent(direction: GalacticDirection, bend: Bend) -> GalacticDirection:
    return GalacticDirection(
        Lean(direction.toward_centre + 2.0 * BEND_REACH * bend.toward_centre),
        Lean(direction.toward_rotation + 2.0 * BEND_REACH * bend.toward_rotation),
        Lean(direction.toward_pole + 2.0 * BEND_REACH * bend.toward_pole),
    )


def _toward_centre(direction: GalacticDirection) -> Share:
    """How far round toward the centre a direction looks: 1 at the centre, 0 opposite it."""
    return Share(0.5 + 0.5 * direction.toward_centre)


def _disc(direction: GalacticDirection) -> Share:
    toward = _toward_centre(direction)
    along = ANTICENTRE + (1.0 - ANTICENTRE) * toward * toward
    thin = _bell(direction.toward_pole, THIN_HEIGHT)
    wide = _bell(direction.toward_pole, WIDE_HEIGHT)
    return Share(along * (THIN_SHARE * thin + (1.0 - THIN_SHARE) * wide))


def _bulge(direction: GalacticDirection) -> Share:
    # Opposite the centre the same sum would draw a second bulge, since it looks only at how far a
    # direction is from the line through the centre.
    if direction.toward_centre <= 0.0:
        return Share(0.0)
    return Share(_bell(direction.toward_rotation, BULGE_LENGTH) * _bell(direction.toward_pole, BULGE_HEIGHT))


def _satellites(direction: GalacticDirection) -> Share:
    share = 0.0
    for satellite in SATELLITES:
        beside = math.fsum(part * other for part, other in zip(_parts(direction), _parts(satellite.toward)))
        # 2 - 2 cos is the square of the straight distance between two directions.
        share += math.exp(-max(2.0 - 2.0 * beside, 0.0) / (satellite.reach * satellite.reach))
    return Share(share)


def _dust(direction: GalacticDirection, bent: GalacticDirection, bend: Bend) -> Share:
    lane = _bell(direction.toward_pole - DUST_WANDER * bend.toward_pole, DUST_HEIGHT)
    filaments = ridged_noise(bent, DUST_FREQUENCY, DUST_OCTAVES, DUST_SEED)
    reach = DUST_ANTICENTRE + (1.0 - DUST_ANTICENTRE) * _toward_centre(direction)
    return Share(DUST_DEPTH * _eased(DUST_FROM, DUST_FULL, Level(lane * filaments)) * reach)


def _nebulae(direction: GalacticDirection) -> Glow:
    # Looked up straight and not at the bent place: a nebula is a round cloud, and the bend draws it out
    # into a streak.
    level = clouded_noise(direction, NEBULA_FREQUENCY, NEBULA_OCTAVES, NEBULA_SEED)
    in_plane = _bell(direction.toward_pole, NEBULA_HEIGHT)
    glowing = _eased(NEBULA_FROM, NEBULA_FULL, level)
    shining = _eased(REFLECTION_FROM, REFLECTION_FULL, level)
    return NEBULA_RIM.blended(NEBULA_MIDDLE, glowing).scaled(glowing * in_plane).plus(
        REFLECTION_COLOUR.scaled(shining * in_plane)
    )


def _sheet_star(direction: GalacticDirection, sheet: StarSheet, crowd: Share, pixel: Reach) -> Glow:
    """The light of the one star that the cell of a sheet may hold, at a place in that cell."""
    steps = tuple((part + _LATTICE_OFFSET) * sheet.frequency for part in _parts(direction))
    whole = tuple(math.floor(step) for step in steps)
    bits = _corner_bits((whole[0], whole[1], whole[2]), sheet.seed)
    if (bits >> _CHANCE_SHIFT) / _BYTE_STEPS >= crowd:
        return DARK
    away = 0.0
    for axis in range(len(steps)):
        middle = STAR_MARGIN + (1.0 - 2.0 * STAR_MARGIN) * ((bits >> (axis * _BYTE_BITS)) & _BYTE_MASK) / _BYTE_STEPS
        away += (steps[axis] - whole[axis] - middle) * (steps[axis] - whole[axis] - middle)
    reach = min(max(STAR_REACH, STAR_PIXELS * pixel) * sheet.frequency, STAR_MARGIN)
    spread = max(1.0 - away / (reach * reach), 0.0)
    more = _mixed(bits)
    chance = (more & _HALF_MASK) / _HALF_STEPS
    light = sheet.light * (STAR_FAINTEST + (1.0 - STAR_FAINTEST) * chance * chance * chance * chance)
    return STAR_HOT.blended(STAR_COOL, Share((more >> _HALF_BITS) / _HALF_STEPS)).scaled(light * spread * spread)


def _faint_stars(direction: GalacticDirection, crowd: Share, pixel: Reach) -> Glow:
    stars = DARK
    for sheet in STAR_SHEETS:
        stars = stars.plus(_sheet_star(direction, sheet, crowd, pixel))
    return stars


@deal.pure
@deal.pre(lambda direction, pixel: is_direction(direction) and pixel >= 0)
@deal.post(_is_glow)
def light_toward(direction: GalacticDirection, pixel: Reach) -> Glow:
    """The light of the Milky Way in one direction, as it reaches the eye at midnight.

    `pixel` is how much of the sky one pixel of the screen covers there, which sets how wide a faint
    star is drawn. The haze comes first: the disc, the bulge and the satellites, each in its colour,
    broken up by the star clouds. Dust then takes its share out of it, and the nebulae shine on top,
    dimmed by the same dust. Last come the faint stars, crowded where the haze is bright.
    """
    disc = _disc(direction)
    bulge = _bulge(direction)
    satellites = _satellites(direction)
    band = disc + bulge + satellites
    if band < FAINTEST_DRAWN:
        return _faint_stars(direction, CROWD_EVERYWHERE, pixel).capped()
    bend = _bend(direction)
    bent = _bent(direction, bend)
    clumps = CLUMP_FLOOR + CLUMP_RANGE * _eased(
        CLUMPS_FROM, CLUMPS_FULL, clouded_noise(direction, CLUMP_FREQUENCY, CLUMP_OCTAVES, CLUMP_SEED)
    )
    dust = _dust(direction, bent, bend)
    haze = (
        DISC_BLUE.blended(DISC_VIOLET, Share(0.5 + bend.toward_centre))
        .scaled(disc)
        .plus(BULGE_COLOUR.scaled(bulge))
        .plus(CORE_COLOUR.scaled(bulge * bulge))
        .plus(SATELLITE_COLOUR.scaled(satellites))
        .scaled(clumps * (MOTTLE_FLOOR + MOTTLE_RANGE * smooth_noise(bent, MOTTLE_FREQUENCY, MOTTLE_SEED)))
        .dimmed(DUST_TAKES.scaled(dust))
    )
    nebulae = _nebulae(direction).scaled(1.0 - dust)
    crowd = Share(min(CROWD_EVERYWHERE + CROWD_IN_BAND * band * clumps * (1.0 - dust), CROWD_MOST))
    return haze.plus(nebulae).plus(_faint_stars(direction, crowd, pixel)).capped()


def _vec3(glow: Glow) -> str:
    return f"vec3({glow.red!r}, {glow.green!r}, {glow.blue!r})"


def _satellite_row(satellite: Satellite) -> str:
    toward = satellite.toward
    return (
        f"    vec4({toward.toward_centre:.6f}, {toward.toward_rotation:.6f}, {toward.toward_pole:.6f}, {satellite.reach!r})"
    )


def _sheet_row(sheet: StarSheet) -> str:
    return f"    vec3({sheet.frequency!r}, {sheet.light!r}, {float(sheet.seed)!r})"


_SATELLITE_ROWS = ",\n".join(_satellite_row(satellite) for satellite in SATELLITES)
_SHEET_ROWS = ",\n".join(_sheet_row(sheet) for sheet in STAR_SHEETS)

# The galaxy in GLSL, with the constants it is written from. A direction is a vec3: toward the centre,
# toward the rotation, toward the pole.
GLSL = GlslCode(f"""\
const float PACK_GALAXY_ANTICENTRE = {ANTICENTRE!r};
const float PACK_GALAXY_THIN_HEIGHT = {THIN_HEIGHT!r};
const float PACK_GALAXY_WIDE_HEIGHT = {WIDE_HEIGHT!r};
const float PACK_GALAXY_THIN_SHARE = {THIN_SHARE!r};
const float PACK_GALAXY_BULGE_LENGTH = {BULGE_LENGTH!r};
const float PACK_GALAXY_BULGE_HEIGHT = {BULGE_HEIGHT!r};
const float PACK_GALAXY_FAINTEST_DRAWN = {FAINTEST_DRAWN!r};
const float PACK_GALAXY_BEND_FREQUENCY = {BEND_FREQUENCY!r};
const float PACK_GALAXY_BEND_REACH = {BEND_REACH!r};
const ivec3 PACK_GALAXY_BEND_SEEDS = ivec3({BEND_SEEDS[0]}, {BEND_SEEDS[1]}, {BEND_SEEDS[2]});
const float PACK_GALAXY_CLUMP_FREQUENCY = {CLUMP_FREQUENCY!r};
const int PACK_GALAXY_CLUMP_OCTAVES = {CLUMP_OCTAVES};
const float PACK_GALAXY_CLUMPS_FROM = {CLUMPS_FROM!r};
const float PACK_GALAXY_CLUMPS_FULL = {CLUMPS_FULL!r};
const float PACK_GALAXY_CLUMP_FLOOR = {CLUMP_FLOOR!r};
const float PACK_GALAXY_CLUMP_RANGE = {CLUMP_RANGE!r};
const int PACK_GALAXY_CLUMP_SEED = {CLUMP_SEED};
const float PACK_GALAXY_MOTTLE_FREQUENCY = {MOTTLE_FREQUENCY!r};
const float PACK_GALAXY_MOTTLE_FLOOR = {MOTTLE_FLOOR!r};
const float PACK_GALAXY_MOTTLE_RANGE = {MOTTLE_RANGE!r};
const int PACK_GALAXY_MOTTLE_SEED = {MOTTLE_SEED};
const float PACK_GALAXY_DUST_WANDER = {DUST_WANDER!r};
const float PACK_GALAXY_DUST_HEIGHT = {DUST_HEIGHT!r};
const float PACK_GALAXY_DUST_FREQUENCY = {DUST_FREQUENCY!r};
const int PACK_GALAXY_DUST_OCTAVES = {DUST_OCTAVES};
const float PACK_GALAXY_DUST_FROM = {DUST_FROM!r};
const float PACK_GALAXY_DUST_FULL = {DUST_FULL!r};
const float PACK_GALAXY_DUST_DEPTH = {DUST_DEPTH!r};
const float PACK_GALAXY_DUST_ANTICENTRE = {DUST_ANTICENTRE!r};
const int PACK_GALAXY_DUST_SEED = {DUST_SEED};
const vec3 PACK_GALAXY_DUST_TAKES = {_vec3(DUST_TAKES)};
const float PACK_GALAXY_NEBULA_FREQUENCY = {NEBULA_FREQUENCY!r};
const int PACK_GALAXY_NEBULA_OCTAVES = {NEBULA_OCTAVES};
const float PACK_GALAXY_NEBULA_FROM = {NEBULA_FROM!r};
const float PACK_GALAXY_NEBULA_FULL = {NEBULA_FULL!r};
const float PACK_GALAXY_NEBULA_HEIGHT = {NEBULA_HEIGHT!r};
const float PACK_GALAXY_REFLECTION_FROM = {REFLECTION_FROM!r};
const float PACK_GALAXY_REFLECTION_FULL = {REFLECTION_FULL!r};
const int PACK_GALAXY_NEBULA_SEED = {NEBULA_SEED};
const vec3 PACK_GALAXY_NEBULA_RIM = {_vec3(NEBULA_RIM)};
const vec3 PACK_GALAXY_NEBULA_MIDDLE = {_vec3(NEBULA_MIDDLE)};
const vec3 PACK_GALAXY_REFLECTION_COLOUR = {_vec3(REFLECTION_COLOUR)};
const vec3 PACK_GALAXY_DISC_BLUE = {_vec3(DISC_BLUE)};
const vec3 PACK_GALAXY_DISC_VIOLET = {_vec3(DISC_VIOLET)};
const vec3 PACK_GALAXY_BULGE_COLOUR = {_vec3(BULGE_COLOUR)};
const vec3 PACK_GALAXY_CORE_COLOUR = {_vec3(CORE_COLOUR)};
const vec3 PACK_GALAXY_SATELLITE_COLOUR = {_vec3(SATELLITE_COLOUR)};
const float PACK_GALAXY_CROWD_EVERYWHERE = {CROWD_EVERYWHERE!r};
const float PACK_GALAXY_CROWD_IN_BAND = {CROWD_IN_BAND!r};
const float PACK_GALAXY_CROWD_MOST = {CROWD_MOST!r};
const float PACK_GALAXY_STAR_MARGIN = {STAR_MARGIN!r};
const float PACK_GALAXY_STAR_PIXELS = {STAR_PIXELS!r};
const float PACK_GALAXY_STAR_REACH = {STAR_REACH!r};
const float PACK_GALAXY_STAR_FAINTEST = {STAR_FAINTEST!r};
const vec3 PACK_GALAXY_STAR_HOT = {_vec3(STAR_HOT)};
const vec3 PACK_GALAXY_STAR_COOL = {_vec3(STAR_COOL)};
const float PACK_GALAXY_LATTICE_OFFSET = {_LATTICE_OFFSET!r};
const float PACK_GALAXY_SMOOTH_OFFSET = {_SMOOTH_OFFSET!r};
const float PACK_GALAXY_SMOOTH_SLOPE = {_SMOOTH_SLOPE!r};
const int PACK_GALAXY_DROPPED_BITS = {_DROPPED_BITS};
const float PACK_GALAXY_LEVEL_STEPS = {_LEVEL_STEPS!r};
const int PACK_GALAXY_BYTE_BITS = {_BYTE_BITS};
const uint PACK_GALAXY_BYTE_MASK = {_BYTE_MASK:#x}U;
const float PACK_GALAXY_BYTE_STEPS = {_BYTE_STEPS!r};
const int PACK_GALAXY_CHANCE_SHIFT = {_CHANCE_SHIFT};
const int PACK_GALAXY_HALF_BITS = {_HALF_BITS};
const uint PACK_GALAXY_HALF_MASK = {_HALF_MASK:#x}U;
const float PACK_GALAXY_HALF_STEPS = {_HALF_STEPS!r};
const int PACK_GALAXY_CELL_CORNERS = {_CELL_CORNERS};
const int PACK_GALAXY_SATELLITE_COUNT = {len(SATELLITES)};
const int PACK_GALAXY_SHEET_COUNT = {len(STAR_SHEETS)};

// Each satellite: the direction of its middle, and its reach.
const vec4 PACK_GALAXY_SATELLITES[PACK_GALAXY_SATELLITE_COUNT] = vec4[PACK_GALAXY_SATELLITE_COUNT](
{_SATELLITE_ROWS}
);

// Each sheet of faint stars: its frequency, the light of its brightest star, and its seed.
const vec3 PACK_GALAXY_SHEETS[PACK_GALAXY_SHEET_COUNT] = vec3[PACK_GALAXY_SHEET_COUNT](
{_SHEET_ROWS}
);

uint packGalaxyMixed(uint bits) {{
    bits ^= bits >> {_MIX_SHIFTS[0]};
    bits *= {_MIX_MULTIPLIERS[0]:#x}U;
    bits ^= bits >> {_MIX_SHIFTS[1]};
    bits *= {_MIX_MULTIPLIERS[1]:#x}U;
    return bits ^ (bits >> {_MIX_SHIFTS[2]});
}}

uint packGalaxyCornerBits(ivec3 corner, int seed) {{
    return packGalaxyMixed(uint(corner.x) * {_FOLD_MULTIPLIERS[0]:#x}U ^ uint(corner.y) * {_FOLD_MULTIPLIERS[1]:#x}U
        ^ uint(corner.z) * {_FOLD_MULTIPLIERS[2]:#x}U ^ uint(seed) * {_FOLD_MULTIPLIERS[3]:#x}U);
}}

float packGalaxyCornerLevel(ivec3 corner, int seed) {{
    return float(packGalaxyCornerBits(corner, seed) >> PACK_GALAXY_DROPPED_BITS) / PACK_GALAXY_LEVEL_STEPS;
}}

float packGalaxyBell(float distance, float reach) {{
    return exp(-(distance / reach) * (distance / reach));
}}

float packGalaxyEased(float low, float high, float value) {{
    float toward = clamp((value - low) / (high - low), 0.0, 1.0);
    return toward * toward * (PACK_GALAXY_SMOOTH_OFFSET - PACK_GALAXY_SMOOTH_SLOPE * toward);
}}

float packGalaxySmoothNoise(vec3 place, float frequency, int seed) {{
    vec3 steps = (place + PACK_GALAXY_LATTICE_OFFSET) * frequency;
    vec3 whole = floor(steps);
    vec3 toward = steps - whole;
    vec3 blend = toward * toward * (PACK_GALAXY_SMOOTH_OFFSET - PACK_GALAXY_SMOOTH_SLOPE * toward);
    float level = 0.0;
    for (int corner = 0; corner < PACK_GALAXY_CELL_CORNERS; corner++) {{
        ivec3 far = ivec3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        vec3 weights = mix(1.0 - blend, blend, vec3(far));
        level += weights.x * weights.y * weights.z * packGalaxyCornerLevel(ivec3(whole) + far, seed);
    }}
    return min(level, 1.0);
}}

float packGalaxyCloudedNoise(vec3 place, float frequency, int octaves, int seed) {{
    float level = 0.0;
    float strength = 1.0;
    float total = 0.0;
    for (int octave = 0; octave < octaves; octave++) {{
        level += strength * packGalaxySmoothNoise(place, frequency * float(1 << octave), seed + octave);
        total += strength;
        strength *= 0.5;
    }}
    return min(level / total, 1.0);
}}

float packGalaxyRidgedNoise(vec3 place, float frequency, int octaves, int seed) {{
    float level = 0.0;
    float strength = 1.0;
    float total = 0.0;
    for (int octave = 0; octave < octaves; octave++) {{
        float ridge = 1.0 - abs(2.0 * packGalaxySmoothNoise(place, frequency * float(1 << octave), seed + octave) - 1.0);
        level += strength * ridge * ridge;
        total += strength;
        strength *= 0.5;
    }}
    return min(level / total, 1.0);
}}

vec3 packGalaxyBend(vec3 direction) {{
    return vec3(
        packGalaxySmoothNoise(direction, PACK_GALAXY_BEND_FREQUENCY, PACK_GALAXY_BEND_SEEDS.x) - 0.5,
        packGalaxySmoothNoise(direction, PACK_GALAXY_BEND_FREQUENCY, PACK_GALAXY_BEND_SEEDS.y) - 0.5,
        packGalaxySmoothNoise(direction, PACK_GALAXY_BEND_FREQUENCY, PACK_GALAXY_BEND_SEEDS.z) - 0.5);
}}

vec3 packGalaxyBent(vec3 direction, vec3 bend) {{
    return direction + 2.0 * PACK_GALAXY_BEND_REACH * bend;
}}

float packGalaxyTowardCentre(vec3 direction) {{
    return 0.5 + 0.5 * direction.x;
}}

float packGalaxyDisc(vec3 direction) {{
    float toward = packGalaxyTowardCentre(direction);
    float along = PACK_GALAXY_ANTICENTRE + (1.0 - PACK_GALAXY_ANTICENTRE) * toward * toward;
    float thin = packGalaxyBell(direction.z, PACK_GALAXY_THIN_HEIGHT);
    float wide = packGalaxyBell(direction.z, PACK_GALAXY_WIDE_HEIGHT);
    return along * (PACK_GALAXY_THIN_SHARE * thin + (1.0 - PACK_GALAXY_THIN_SHARE) * wide);
}}

float packGalaxyBulge(vec3 direction) {{
    if (direction.x <= 0.0) {{
        return 0.0;
    }}
    return packGalaxyBell(direction.y, PACK_GALAXY_BULGE_LENGTH) * packGalaxyBell(direction.z, PACK_GALAXY_BULGE_HEIGHT);
}}

float packGalaxySatellites(vec3 direction) {{
    float share = 0.0;
    for (int satellite = 0; satellite < PACK_GALAXY_SATELLITE_COUNT; satellite++) {{
        float beside = dot(direction, PACK_GALAXY_SATELLITES[satellite].xyz);
        float reach = PACK_GALAXY_SATELLITES[satellite].w;
        share += exp(-max(2.0 - 2.0 * beside, 0.0) / (reach * reach));
    }}
    return share;
}}

float packGalaxyDust(vec3 direction, vec3 bent, vec3 bend) {{
    float lane = packGalaxyBell(direction.z - PACK_GALAXY_DUST_WANDER * bend.z, PACK_GALAXY_DUST_HEIGHT);
    float filaments = packGalaxyRidgedNoise(bent, PACK_GALAXY_DUST_FREQUENCY, PACK_GALAXY_DUST_OCTAVES, PACK_GALAXY_DUST_SEED);
    float reach = PACK_GALAXY_DUST_ANTICENTRE + (1.0 - PACK_GALAXY_DUST_ANTICENTRE) * packGalaxyTowardCentre(direction);
    return PACK_GALAXY_DUST_DEPTH * packGalaxyEased(PACK_GALAXY_DUST_FROM, PACK_GALAXY_DUST_FULL, lane * filaments) * reach;
}}

vec3 packGalaxyNebulae(vec3 direction) {{
    float level = packGalaxyCloudedNoise(direction, PACK_GALAXY_NEBULA_FREQUENCY, PACK_GALAXY_NEBULA_OCTAVES, PACK_GALAXY_NEBULA_SEED);
    float inPlane = packGalaxyBell(direction.z, PACK_GALAXY_NEBULA_HEIGHT);
    float glowing = packGalaxyEased(PACK_GALAXY_NEBULA_FROM, PACK_GALAXY_NEBULA_FULL, level);
    float shining = packGalaxyEased(PACK_GALAXY_REFLECTION_FROM, PACK_GALAXY_REFLECTION_FULL, level);
    return mix(PACK_GALAXY_NEBULA_RIM, PACK_GALAXY_NEBULA_MIDDLE, glowing) * (glowing * inPlane)
        + PACK_GALAXY_REFLECTION_COLOUR * (shining * inPlane);
}}

vec3 packGalaxySheetStar(vec3 direction, vec3 sheet, float crowd, float pixel) {{
    vec3 steps = (direction + PACK_GALAXY_LATTICE_OFFSET) * sheet.x;
    vec3 whole = floor(steps);
    uint bits = packGalaxyCornerBits(ivec3(whole), int(sheet.z));
    if (float(bits >> PACK_GALAXY_CHANCE_SHIFT) / PACK_GALAXY_BYTE_STEPS >= crowd) {{
        return vec3(0.0);
    }}
    vec3 middle = PACK_GALAXY_STAR_MARGIN + (1.0 - 2.0 * PACK_GALAXY_STAR_MARGIN) * vec3(
        float(bits & PACK_GALAXY_BYTE_MASK),
        float((bits >> PACK_GALAXY_BYTE_BITS) & PACK_GALAXY_BYTE_MASK),
        float((bits >> (2 * PACK_GALAXY_BYTE_BITS)) & PACK_GALAXY_BYTE_MASK)) / PACK_GALAXY_BYTE_STEPS;
    vec3 from = steps - whole - middle;
    float away = dot(from, from);
    float reach = min(max(PACK_GALAXY_STAR_REACH, PACK_GALAXY_STAR_PIXELS * pixel) * sheet.x, PACK_GALAXY_STAR_MARGIN);
    float spread = max(1.0 - away / (reach * reach), 0.0);
    uint more = packGalaxyMixed(bits);
    float chance = float(more & PACK_GALAXY_HALF_MASK) / PACK_GALAXY_HALF_STEPS;
    float light = sheet.y * (PACK_GALAXY_STAR_FAINTEST
        + (1.0 - PACK_GALAXY_STAR_FAINTEST) * chance * chance * chance * chance);
    return mix(PACK_GALAXY_STAR_HOT, PACK_GALAXY_STAR_COOL, float(more >> PACK_GALAXY_HALF_BITS) / PACK_GALAXY_HALF_STEPS)
        * (light * spread * spread);
}}

vec3 packGalaxyFaintStars(vec3 direction, float crowd, float pixel) {{
    vec3 stars = vec3(0.0);
    for (int sheet = 0; sheet < PACK_GALAXY_SHEET_COUNT; sheet++) {{
        stars += packGalaxySheetStar(direction, PACK_GALAXY_SHEETS[sheet], crowd, pixel);
    }}
    return stars;
}}

vec3 packGalaxyLight(vec3 direction, float pixel) {{
    float disc = packGalaxyDisc(direction);
    float bulge = packGalaxyBulge(direction);
    float satellites = packGalaxySatellites(direction);
    float band = disc + bulge + satellites;
    if (band < PACK_GALAXY_FAINTEST_DRAWN) {{
        return min(packGalaxyFaintStars(direction, PACK_GALAXY_CROWD_EVERYWHERE, pixel), 1.0);
    }}
    vec3 bend = packGalaxyBend(direction);
    vec3 bent = packGalaxyBent(direction, bend);
    float clumps = PACK_GALAXY_CLUMP_FLOOR + PACK_GALAXY_CLUMP_RANGE * packGalaxyEased(
        PACK_GALAXY_CLUMPS_FROM, PACK_GALAXY_CLUMPS_FULL, packGalaxyCloudedNoise(
            direction, PACK_GALAXY_CLUMP_FREQUENCY, PACK_GALAXY_CLUMP_OCTAVES, PACK_GALAXY_CLUMP_SEED));
    float dust = packGalaxyDust(direction, bent, bend);
    vec3 haze = (mix(PACK_GALAXY_DISC_BLUE, PACK_GALAXY_DISC_VIOLET, 0.5 + bend.x) * disc
        + PACK_GALAXY_BULGE_COLOUR * bulge + PACK_GALAXY_CORE_COLOUR * (bulge * bulge)
        + PACK_GALAXY_SATELLITE_COLOUR * satellites)
        * (clumps * (PACK_GALAXY_MOTTLE_FLOOR + PACK_GALAXY_MOTTLE_RANGE * packGalaxySmoothNoise(
            bent, PACK_GALAXY_MOTTLE_FREQUENCY, PACK_GALAXY_MOTTLE_SEED)))
        * (1.0 - PACK_GALAXY_DUST_TAKES * dust);
    vec3 nebulae = packGalaxyNebulae(direction) * (1.0 - dust);
    float crowd = min(PACK_GALAXY_CROWD_EVERYWHERE + PACK_GALAXY_CROWD_IN_BAND * band * clumps * (1.0 - dust),
        PACK_GALAXY_CROWD_MOST);
    return min(haze + nebulae + packGalaxyFaintStars(direction, crowd, pixel), 1.0);
}}

"""
)
