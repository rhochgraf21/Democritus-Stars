"""Noise over the world: a level from 0 to 1 for every place of the cloud layer, the same each time.

A lattice lies over the world, and a hash of which corner it is gives each corner bits of its own. A
lattice is always a power of two of blocks wide, so that the corner a place belongs to comes from shifting
bits, which GLSL defines for a negative number where it does not define division. A place is kept as
whole blocks and a rest, so that it stays exact millions of blocks from the origin, where one float would
not.

Two noises are made of a lattice. Smooth noise gives each corner a level and blends the levels between
corners. Billow noise puts one round puff in each cell, its centre placed by the bits, and is how near the
nearest centre is.

The maths lives twice: as the functions here, which the tests check, and as GLSL, which the game runs.
Each GLSL function mirrors the Python one of the same name line for line. The hashes are 32 bit integer
arithmetic and agree to the bit; the rest is float arithmetic and differs by rounding.
"""

import math
from dataclasses import dataclass
from typing import NewType

import deal

from shader.effects import GlslCode

# A whole number of blocks along one axis of the world.
BlockCount = NewType("BlockCount", int)

# A length in blocks.
BlockSpan = NewType("BlockSpan", float)

# The size of a lattice along one axis: its corners are 2 ** exponent blocks apart.
SizeExponent = NewType("SizeExponent", int)

# Which corner of a lattice along one axis, counted from the origin of the world.
CornerIndex = NewType("CornerIndex", int)

# The bits kept of a corner's index from east to west. All bits kept is a lattice that never repeats.
WrapMask = NewType("WrapMask", int)

# What tells one noise from another over the same lattice.
Seed = NewType("Seed", int)

# 32 bits, as a hash takes and gives them.
Bits = NewType("Bits", int)

# A value of noise, or a point on its range: 0 to 1.
Level = NewType("Level", float)

# How much of something: 0 is none of it, 1 is all of it.
Share = NewType("Share", float)


@dataclass(frozen=True)
class LayerPlace:
    """A place in the cloud layer seen from above: whole blocks of the world, and the rest in blocks."""

    east_block: BlockCount
    south_block: BlockCount
    east_rest: BlockSpan
    south_rest: BlockSpan


@dataclass(frozen=True)
class Lattice:
    """How far apart the corners of a lattice are from east to west and from north to south."""

    east: SizeExponent
    south: SizeExponent


@dataclass(frozen=True)
class Corner:
    """One corner of a lattice."""

    east: CornerIndex
    south: CornerIndex


@dataclass(frozen=True)
class PuffCentre:
    """Where the centre of the puff of one lattice cell lies in the cell: 0 at its west or north edge, 1 at the other."""

    east: Share
    south: Share

# The lowest level a noise has.
LOWEST = Level(0.0)

# The level halfway up, which is the average of smooth noise.
MIDDLE = Level(0.5)

# The highest level a noise has.
HIGHEST = Level(1.0)

# None of something.
NONE = Share(0.0)

# Half of something.
HALF = Share(0.5)

# All of something.
ALL = Share(1.0)

# A lattice that does not drift keeps every bit of a corner's index, and so never repeats.
NO_WRAP = WrapMask(-1)

# The two numbers of the smoothstep curve t * t * (3 - 2 * t), which is level at both ends: blended with
# it, noise has no crease along the lines of its lattice.
_SMOOTH_OFFSET = 3.0

_SMOOTH_SLOPE = 2.0

# The 32 bit mixer "lowbias32" of Chris Wellons' hash prospector: three shifts and two multipliers found
# by search to make every input bit change every output bit about half the time.
_WORD = Bits(0xFFFFFFFF)

_MIX_SHIFTS = (16, 15, 16)

_MIX_MULTIPLIERS = (Bits(0x7FEB352D), Bits(0x846CA68B))

# A level is the top 24 bits of a hash, because a 32 bit float holds 24 bits exactly.
_LEVEL_BITS = 24

_DROPPED_BITS = 32 - _LEVEL_BITS

_LEVEL_STEPS = float(1 << _LEVEL_BITS)

# The centre of a puff takes half of those bits each way: one part in 4096 of a cell, finer than a pixel.
_CENTRE_BITS = _LEVEL_BITS // 2

_CENTRE_MASK = Bits((1 << _CENTRE_BITS) - 1)

_CENTRE_STEPS = float(1 << _CENTRE_BITS)

# The level of billow noise averaged over the world: 0.779 over 40000 places, for three seeds alike. It is
# what a lattice of billows too far away to draw is left at, so that the amount of cloud does not change
# with the distance. The middle level, which is the average of smooth noise, made rings in the sky around
# the player where each lattice was left out.
BILLOW_MEAN = Level(0.78)

# A puff reaches one cell from its centre. The puffs that can reach a place are those of the cell the place
# is in and of the cells around it, one step away each way.
_PUFF_REACH = 1.0

_NEIGHBOUR_STEPS = (-1, 0, 1)

# The widest a lattice may be, as a power of two of blocks: 2048 blocks.
LARGEST_LATTICE = SizeExponent(11)

# Beyond this many blocks from the camera's block the game draws no cloud, and a rest is never larger.
FARTHEST_REST = BlockSpan(float(1 << 20))


@deal.pure
@deal.ensure(lambda place, result: result == (max(abs(place.east_rest), abs(place.south_rest)) <= FARTHEST_REST))
def is_placed(place: LayerPlace) -> bool:
    """Whether the rest of a place is no further from its block than the game draws clouds."""
    return abs(place.east_rest) <= FARTHEST_REST and abs(place.south_rest) <= FARTHEST_REST


@deal.pure
@deal.ensure(lambda lattice, result: result == (0 <= min(lattice.east, lattice.south) and max(lattice.east, lattice.south) <= LARGEST_LATTICE))
def is_lattice(lattice: Lattice) -> bool:
    """Whether a lattice is at least a block and at most LARGEST_LATTICE wide each way."""
    return 0 <= lattice.east <= LARGEST_LATTICE and 0 <= lattice.south <= LARGEST_LATTICE


@deal.pure
@deal.ensure(lambda level, result: result == (LOWEST <= level <= HIGHEST))
def is_level(level: Level) -> bool:
    """Whether a number lies from the lowest level to the highest, which a number that is not a number does not."""
    return LOWEST <= level <= HIGHEST


@deal.pure
@deal.pre(lambda bits: 0 <= bits <= _WORD)
@deal.post(lambda result: 0 <= result <= _WORD)
def mixed(bits: Bits) -> Bits:
    """32 bits stirred so that every bit of the input changes about half the bits of the result."""
    bits = Bits(bits ^ (bits >> _MIX_SHIFTS[0]))
    bits = Bits(bits * _MIX_MULTIPLIERS[0] & _WORD)
    bits = Bits(bits ^ (bits >> _MIX_SHIFTS[1]))
    bits = Bits(bits * _MIX_MULTIPLIERS[1] & _WORD)
    return Bits(bits ^ (bits >> _MIX_SHIFTS[2]))


def _stirred(corner: Corner, seed: Seed) -> Bits:
    """The bits that belong to a corner of a lattice: the same for the same corner and seed, each time."""
    return mixed(Bits(mixed(Bits(mixed(Bits(seed & _WORD)) ^ (corner.east & _WORD))) ^ (corner.south & _WORD)))


@deal.pure
@deal.post(lambda result: LOWEST <= result < HIGHEST)
def corner_level(corner: Corner, seed: Seed) -> Level:
    """The level of smooth noise at a corner of a lattice."""
    return Level((_stirred(corner, seed) >> _DROPPED_BITS) / _LEVEL_STEPS)


@deal.pure
@deal.post(lambda result: NONE <= result.east < ALL and NONE <= result.south < ALL)
def puff_centre(corner: Corner, seed: Seed) -> PuffCentre:
    """Where the puff of the lattice cell that starts at a corner has its centre, in the cell."""
    kept = _stirred(corner, seed) >> _DROPPED_BITS
    return PuffCentre(Share((kept >> _CENTRE_BITS) / _CENTRE_STEPS), Share((kept & _CENTRE_MASK) / _CENTRE_STEPS))


@deal.pure
@deal.pre(lambda low, high, value: low < high)
# The two ends are numbers that have an end, and the value is a number: GLSL leaves the rest undefined too.
@deal.pre(lambda low, high, value: math.isfinite(low) and math.isfinite(high) and not math.isnan(value))
@deal.post(is_level)
@deal.ensure(lambda low, high, value, result: value > low or result == LOWEST)
@deal.ensure(lambda low, high, value, result: value < high or result == HIGHEST)
def smoothstep(low: Level, high: Level, value: Level) -> Level:
    """GLSL's smoothstep: nothing up to `low`, all from `high` on, and level at both ends between."""
    t = min(max((value - low) / (high - low), LOWEST), HIGHEST)
    return Level(t * t * (_SMOOTH_OFFSET - _SMOOTH_SLOPE * t))


def _along(block: BlockCount, rest: BlockSpan, exponent: SizeExponent) -> tuple[CornerIndex, Share]:
    """Along one axis: the corner at or before a place, and how far the place is toward the next corner."""
    size = 1 << exponent
    steps = ((block & (size - 1)) + rest) / size
    whole = math.floor(steps)
    return CornerIndex((block >> exponent) + whole), Share(steps - whole)


@deal.pure
@deal.pre(lambda place, lattice, wrap, seed: is_placed(place) and is_lattice(lattice))
@deal.post(is_level)
def value_noise(place: LayerPlace, lattice: Lattice, wrap: WrapMask, seed: Seed) -> Level:
    """The noise of one lattice at a place: the levels of the four corners around it, blended smoothly.

    `wrap` is the bits kept of a corner's index from east to west: the noise repeats that often.
    """
    east, toward_east = _along(place.east_block, place.east_rest, lattice.east)
    south, toward_south = _along(place.south_block, place.south_rest, lattice.south)
    blend_east = toward_east * toward_east * (_SMOOTH_OFFSET - _SMOOTH_SLOPE * toward_east)
    blend_south = toward_south * toward_south * (_SMOOTH_OFFSET - _SMOOTH_SLOPE * toward_south)
    near = CornerIndex(east & wrap)
    far = CornerIndex((east + 1) & wrap)
    north_edge = corner_level(Corner(near, south), seed) * (ALL - blend_east) + corner_level(
        Corner(far, south), seed
    ) * blend_east
    south_edge = corner_level(Corner(near, CornerIndex(south + 1)), seed) * (ALL - blend_east) + corner_level(
        Corner(far, CornerIndex(south + 1)), seed
    ) * blend_east
    return Level(north_edge * (ALL - blend_south) + south_edge * blend_south)


@deal.pure
@deal.pre(lambda place, lattice, wrap, seed: is_placed(place) and is_lattice(lattice))
@deal.post(is_level)
def billow_noise(place: LayerPlace, lattice: Lattice, wrap: WrapMask, seed: Seed) -> Level:
    """The noise of round puffs, one to a cell of a lattice: 1 at the centre of a puff, less further from it.

    The level falls with the square of how far the nearest centre is, which leaves the middle of a puff
    rounded and not a point. It is nothing a whole cell away from every centre, so it has no step where
    a place passes from one cell to the next and other puffs come into reach.
    """
    east, toward_east = _along(place.east_block, place.east_rest, lattice.east)
    south, toward_south = _along(place.south_block, place.south_rest, lattice.south)
    nearest = _PUFF_REACH * _PUFF_REACH
    for step_south in _NEIGHBOUR_STEPS:
        for step_east in _NEIGHBOUR_STEPS:
            centre = puff_centre(Corner(CornerIndex((east + step_east) & wrap), CornerIndex(south + step_south)), seed)
            away_east = step_east + centre.east - toward_east
            away_south = step_south + centre.south - toward_south
            nearest = min(nearest, away_east * away_east + away_south * away_south)
    return Level(_PUFF_REACH * _PUFF_REACH - nearest)

# The noise in GLSL, with the constants it is written from. Each function is the Python function of the same
# name, line for line. Change one and change the other.
GLSL = GlslCode(f"""\
const float PACK_MIDDLE = {MIDDLE!r};
const float PACK_NONE = {NONE!r};
const float PACK_HALF = {HALF!r};
const float PACK_ALL = {ALL!r};
const int PACK_NO_WRAP = {NO_WRAP};
const float PACK_LOWEST = {LOWEST!r};
const float PACK_HIGHEST = {HIGHEST!r};
const float PACK_SMOOTH_OFFSET = {_SMOOTH_OFFSET!r};
const float PACK_SMOOTH_SLOPE = {_SMOOTH_SLOPE!r};
const int PACK_DROPPED_BITS = {_DROPPED_BITS};
const float PACK_LEVEL_STEPS = {_LEVEL_STEPS!r};
const int PACK_CENTRE_BITS = {_CENTRE_BITS};
const uint PACK_CENTRE_MASK = {_CENTRE_MASK:#x}U;
const float PACK_CENTRE_STEPS = {_CENTRE_STEPS!r};
const float PACK_PUFF_REACH = {_PUFF_REACH!r};
const float PACK_BILLOW_MEAN = {BILLOW_MEAN!r};
const int PACK_FIRST_NEIGHBOUR = {_NEIGHBOUR_STEPS[0]};
const int PACK_LAST_NEIGHBOUR = {_NEIGHBOUR_STEPS[-1]};

uint packMixed(uint bits) {{
    bits ^= bits >> {_MIX_SHIFTS[0]};
    bits *= {_MIX_MULTIPLIERS[0]:#x}U;
    bits ^= bits >> {_MIX_SHIFTS[1]};
    bits *= {_MIX_MULTIPLIERS[1]:#x}U;
    return bits ^ (bits >> {_MIX_SHIFTS[2]});
}}

uint packStirred(ivec2 corner, int seed) {{
    return packMixed(packMixed(packMixed(uint(seed)) ^ uint(corner.x)) ^ uint(corner.y));
}}

float packCornerLevel(ivec2 corner, int seed) {{
    return float(packStirred(corner, seed) >> PACK_DROPPED_BITS) / PACK_LEVEL_STEPS;
}}

vec2 packPuffCentre(ivec2 corner, int seed) {{
    uint kept = packStirred(corner, seed) >> PACK_DROPPED_BITS;
    return vec2(float(kept >> PACK_CENTRE_BITS), float(kept & PACK_CENTRE_MASK)) / PACK_CENTRE_STEPS;
}}

// `block` and `rest` are a place: whole blocks of the world east and south, and the rest in blocks.
float packValueNoise(ivec2 block, vec2 rest, ivec2 lattice, int wrap, int seed) {{
    ivec2 size = ivec2(1) << lattice;
    vec2 steps = (vec2(block & (size - 1)) + rest) / vec2(size);
    vec2 whole = floor(steps);
    ivec2 corner = (block >> lattice) + ivec2(whole);
    vec2 toward = steps - whole;
    vec2 blend = toward * toward * (PACK_SMOOTH_OFFSET - PACK_SMOOTH_SLOPE * toward);
    int near = corner.x & wrap;
    int far = (corner.x + 1) & wrap;
    float northEdge = packCornerLevel(ivec2(near, corner.y), seed) * (PACK_ALL - blend.x)
        + packCornerLevel(ivec2(far, corner.y), seed) * blend.x;
    float southEdge = packCornerLevel(ivec2(near, corner.y + 1), seed) * (PACK_ALL - blend.x)
        + packCornerLevel(ivec2(far, corner.y + 1), seed) * blend.x;
    return northEdge * (PACK_ALL - blend.y) + southEdge * blend.y;
}}

float packBillowNoise(ivec2 block, vec2 rest, ivec2 lattice, int wrap, int seed) {{
    ivec2 size = ivec2(1) << lattice;
    vec2 steps = (vec2(block & (size - 1)) + rest) / vec2(size);
    vec2 whole = floor(steps);
    ivec2 corner = (block >> lattice) + ivec2(whole);
    vec2 toward = steps - whole;
    float nearest = PACK_PUFF_REACH * PACK_PUFF_REACH;
    for (int stepSouth = PACK_FIRST_NEIGHBOUR; stepSouth <= PACK_LAST_NEIGHBOUR; stepSouth++) {{
        for (int stepEast = PACK_FIRST_NEIGHBOUR; stepEast <= PACK_LAST_NEIGHBOUR; stepEast++) {{
            vec2 centre = packPuffCentre(ivec2((corner.x + stepEast) & wrap, corner.y + stepSouth), seed);
            vec2 away = vec2(stepEast, stepSouth) + centre - toward;
            nearest = min(nearest, dot(away, away));
        }}
    }}
    return PACK_PUFF_REACH * PACK_PUFF_REACH - nearest;
}}
""")
