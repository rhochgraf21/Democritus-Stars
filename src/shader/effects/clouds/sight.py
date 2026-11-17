"""What a line of sight meets in the cloud layer, how much of what is behind it that hides, and how it is lit.

The layer is a slab SLAB_HEIGHT blocks thick. A cloud stands in it from a flat base at the bottom up to
its top (shader.effects.clouds.forms). A line of sight enters the slab at a face, from below or from above,
and is followed through the slab until it is under a top (seen_cloud). From below that shows the bases
and, through the gaps, the sides of the clouds behind; from above, the tops.

Light comes from above. The base of a thick cloud takes the game's colour of an underside, and the colour
rises to the game's colour of a top with the height at which the cloud is seen. A thin part of a cloud
lets the light through, so it takes the top's colour at any height.

The maths lives twice, as in shader.effects.clouds.noise.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import NewType

import deal

from shader.effects import GlslCode
from shader.effects.clouds.forms import (
    CLOCK_END,
    CLOCK_START,
    FULL_CLOUD,
    NO_CLOUD,
    NO_GAIN,
    Clock,
    Column,
    Density,
    Footprint,
    Gain,
    layer_column,
    layer_height,
    place_kind,
    tallest_top,
)
from shader.effects.clouds.noise import (
    ALL,
    FARTHEST_REST,
    HALF,
    NONE,
    BlockSpan,
    LayerPlace,
    Level,
    Share,
    is_placed,
    smoothstep,
)

# How much of what lies behind a fragment it hides: 0 for none, 1 for all.
Alpha = NewType("Alpha", float)

# How many times a line of sight is looked at on its way through the slab.
StepCount = NewType("StepCount", int)


class Side(Enum):
    """The side of the slab that a line of sight enters it from."""

    BELOW = "below"
    ABOVE = "above"


@dataclass(frozen=True)
class Travel:
    """How far a line of sight moves over the layer, east and south, while it crosses the slab."""

    east: BlockSpan
    south: BlockSpan


@dataclass(frozen=True)
class Seen:
    """What a line of sight meets in the slab: at what height, and how much cloud it looks into there."""

    height: Share
    density: Density

CLEAR = Alpha(0.0)

OPAQUE = Alpha(1.0)

# The most a cloud hides where it is thickest. The game's clouds are 0.8. A wholly opaque cloud lets
# nothing through, so every density above none would hide everything and the soft edge would be a hard
# one. Held just under opaque, the edge stays soft whatever colour the game gives.
DENSEST_ALPHA = Alpha(0.95)

# Under this a pixel is not drawn. It is the alpha under which the game's own first phase of
# order-independent transparency drops a fragment (oit.glsl).
FAINTEST_ALPHA = Alpha(0.01)

# How thick the slab is that the clouds stand in, from their flat bases up. The game's layer is 4 blocks.
# 64 lets a cumulus stand about as tall as its billows are wide, which is what makes it a heap.
SLAB_HEIGHT = BlockSpan(64.0)

# A line of sight is looked at this many times across the slab. Fewer, and it steps over a narrow cloud.
MARCH_STEPS = StepCount(10)

NO_STEPS = StepCount(0)

# Once a line of sight is known to go under a top between two looks, the stretch between is halved this
# many times. Without it the side of a cloud is drawn as a stair, one tread to a look.
HALVINGS = StepCount(4)

# A cloud whose top stands this high in the slab or higher is as much cloud to look through as cloud
# gets. Lower, toward its rim, it is seen through.
SOLID_HEIGHT = Share(0.25)


# The furthest a line of sight is followed over the layer. Nearly level, it would cross the whole sky.
LONGEST_TRAVEL = BlockSpan(SLAB_HEIGHT * 16)

# Keeps a drop of nothing, which a camera level with a face has, from being divided by.
_SMALLEST_DROP = BlockSpan(1e-3)

# From this height in the slab up, the side of a cloud is in full light. Under it the cloud's own bulk
# shades it, down to the base, which gets none.
LIT_HEIGHT = Share(0.3)

# How far a length may go over its limit by rounding.
_ROUNDING = 1e-9


@deal.pure
@deal.pre(lambda east, south, drop: abs(east) <= FARTHEST_REST and abs(south) <= FARTHEST_REST and drop >= 0)
@deal.post(lambda result: math.hypot(result.east, result.south) <= LONGEST_TRAVEL * (ALL + _ROUNDING))
def sight_travel(east: BlockSpan, south: BlockSpan, drop: BlockSpan) -> Travel:
    """How far a line of sight moves over the layer while it crosses the slab.

    `east` and `south` are where the line enters the slab, from the camera, and `drop` is how far the
    camera is below or above that face. The line goes on as it came, so it moves over the layer by its
    slope times the height of the slab, and by no more than LONGEST_TRAVEL.
    """
    slope = SLAB_HEIGHT / max(drop, _SMALLEST_DROP)
    length = math.hypot(east * slope, south * slope)
    held = min(ALL, LONGEST_TRAVEL / max(length, _SMALLEST_DROP))
    return Travel(BlockSpan(east * slope * held), BlockSpan(south * slope * held))


def _ray_height(side: Side, progress: Share) -> Share:
    """How high in the slab a line of sight is when it is this far through: it rises from below, sinks from above."""
    return progress if side is Side.BELOW else Share(ALL - progress)


def _is_under(
    entry: LayerPlace, travel: Travel, side: Side, progress: Share, kind: Level, clock: Clock, footprint: Footprint
) -> bool:
    """Whether a line of sight, this far through the slab, is under the top of the cloud over where it is."""
    return _ray_height(side, progress) < layer_height(_along_sight(entry, travel, progress), kind, clock, footprint)


def _along_sight(entry: LayerPlace, travel: Travel, progress: Share) -> LayerPlace:
    return LayerPlace(
        entry.east_block,
        entry.south_block,
        BlockSpan(entry.east_rest + travel.east * progress),
        BlockSpan(entry.south_rest + travel.south * progress),
    )


@deal.pure
@deal.pre(lambda entry, travel, side, steps, clock, footprint: is_placed(entry) and steps >= 0 and footprint >= 0)
@deal.pre(lambda entry, travel, side, steps, clock, footprint: CLOCK_START <= clock <= CLOCK_END)
@deal.pre(lambda entry, travel, side, steps, clock, footprint: math.hypot(travel.east, travel.south) <= FARTHEST_REST)
@deal.post(lambda result: NONE <= result.height <= ALL)
@deal.post(lambda result: NO_CLOUD <= result.density <= FULL_CLOUD)
def seen_cloud(
    entry: LayerPlace, travel: Travel, side: Side, steps: StepCount, clock: Clock, footprint: Footprint
) -> Seen:
    """What a line of sight meets on its way through the slab, or clear sky at height 0 if it meets nothing.

    A cloud stands over a place from the bottom of the slab, where all of it is, up to the height of its
    top. From below, the line meets the base of the cloud over the place where it enters, and where that
    is as much cloud as cloud gets, nothing behind it shows. Otherwise the line is looked at `steps` times,
    evenly, until it is under the top of the cloud over the place it has reached, and the stretch since
    the look before is halved HALVINGS times to find where it went under; what is seen is the densest of
    the base, the cloud there and the cloud one look further, which the line goes on into. A line from above that goes under no top meets the bottom of the slab. With no steps the layer
    is drawn flat: the cloud over the place of entry, its base from below and its top from above.

    Every place along the line is given the kind of the place of entry: the kind changes over hundreds of
    blocks, and working it out at every look would cost as much as the looks.
    """
    kind = place_kind(entry)
    column = layer_column(entry, kind, clock, footprint)
    if steps == NO_STEPS:
        return Seen(column.height if side is Side.ABOVE else NONE, looked_into(column))
    base = looked_into(column) if side is Side.BELOW else NO_CLOUD
    if base >= FULL_CLOUD:
        return Seen(NONE, base)
    highest = tallest_top(kind)
    clear_until = NONE
    for step in range(1, steps + 1):
        under_from = Share(step / steps)
        if _ray_height(side, under_from) < highest and _is_under(
            entry, travel, side, under_from, kind, clock, footprint
        ):
            for _ in range(HALVINGS):
                middle = Share((clear_until + under_from) * HALF)
                if _is_under(entry, travel, side, middle, kind, clock, footprint):
                    under_from = middle
                else:
                    clear_until = middle
            # Where the line goes under a top it is at the cloud's rim, and the cloud it goes on into
            # is one look further.
            further = Share(min(under_from + ALL / steps, ALL))
            met = layer_column(_along_sight(entry, travel, under_from), kind, clock, footprint)
            beyond = layer_column(_along_sight(entry, travel, further), kind, clock, footprint)
            return Seen(_ray_height(side, under_from), Density(max(base, looked_into(met), looked_into(beyond))))
        clear_until = under_from
    if side is Side.ABOVE:
        return Seen(NONE, looked_into(layer_column(_along_sight(entry, travel, ALL), kind, clock, footprint)))
    return Seen(NONE, base)


@deal.pure
@deal.pre(lambda column: NONE <= column.height <= ALL and NO_CLOUD <= column.density <= FULL_CLOUD)
@deal.post(lambda result: NO_CLOUD <= result <= FULL_CLOUD)
@deal.ensure(lambda column, result: result >= column.density)
@deal.ensure(lambda column, result: column.height < SOLID_HEIGHT or result == FULL_CLOUD)
def looked_into(column: Column) -> Density:
    """How much cloud a line of sight looks into where it meets the cloud over a place.

    The density of a column is what a flat cloud would have there. A cloud with a top that stands high
    is that much deeper to look through, and from SOLID_HEIGHT up it is as much cloud as cloud gets. What
    is seen through is what the flat cloud and the depth each let through, one after the other, and the
    depth comes in level at both ends: a crease anywhere here shows as a line around every cloud.
    """
    deep = smoothstep(Level(NONE), Level(SOLID_HEIGHT), Level(column.height))
    # Never less than the flat cloud: for a density too small to tell from none beside 1, taking it from
    # 1 and back loses it, and the sum alone would give no cloud where there is a trace of one.
    return Density(max(column.density, FULL_CLOUD - (FULL_CLOUD - column.density) * (ALL - deep)))


@deal.pure
@deal.pre(lambda base, density: CLEAR <= base <= OPAQUE)
@deal.pre(lambda base, density: NO_CLOUD <= density <= FULL_CLOUD)
@deal.post(lambda result: CLEAR <= result <= DENSEST_ALPHA)
@deal.ensure(lambda base, density, result: density > NO_CLOUD or result == CLEAR)
def cloud_alpha(base: Alpha, density: Density) -> Alpha:
    """How much a cloud hides where it has this density, when the game's cloud colour hides `base`.

    What a slab of cloud lets through is multiplied when slabs are stacked, so a share of the density
    is a power of what the whole lets through, not a share of the alpha.
    """
    return Alpha(OPAQUE - (OPAQUE - min(base, DENSEST_ALPHA)) ** density)


@deal.pure
@deal.pre(lambda height, density: NONE <= height <= ALL and NO_CLOUD <= density <= FULL_CLOUD)
@deal.post(lambda result: NONE <= result <= ALL)
@deal.ensure(lambda height, density, result: height < LIT_HEIGHT or result == ALL)
@deal.ensure(lambda height, density, result: density > NO_CLOUD or result == ALL)
def lift(height: Share, density: Density) -> Share:
    """How much light a seen part of a cloud gets: more the higher it stands, and more the thinner it is.

    Light comes from above, so the base of a thick cloud is its darkest part, and from LIT_HEIGHT up
    nothing of the cloud stands in the way. A thin part lets the light through wherever it is. The shade
    is the cloud's density times how far under LIT_HEIGHT it is seen, which comes in level at both ends.
    """
    lit = smoothstep(Level(NONE), Level(LIT_HEIGHT), Level(height))
    return Share(ALL - density * (ALL - lit))


@deal.pure
@deal.pre(lambda shade, lit: NONE <= shade <= NO_GAIN and NONE <= lit <= ALL)
@deal.post(lambda result: result <= NO_GAIN)
@deal.ensure(lambda shade, lit, result: result >= shade)
@deal.ensure(lambda shade, lit, result: lit < ALL or result == NO_GAIN)
def light_gain(shade: Gain, lit: Share) -> Gain:
    """What the game's colour of a cloud top is multiplied by: `shade` with no light, 1 with all of it."""
    return Gain(shade * (ALL - lit) + NO_GAIN * lit)

# The line of sight in GLSL, with the constants it is written from, to go after the forms. Each function is
# the Python function of the same name, line for line. Change one and change the other.
GLSL = GlslCode(f"""\
const float PACK_OPAQUE = {OPAQUE!r};
const float PACK_DENSEST_ALPHA = {DENSEST_ALPHA!r};
const float PACK_FAINTEST_ALPHA = {FAINTEST_ALPHA!r};
const float PACK_SLAB_HEIGHT = {SLAB_HEIGHT!r};
const int PACK_MARCH_STEPS = {MARCH_STEPS};
const int PACK_NO_STEPS = {NO_STEPS};
const float PACK_LONGEST_TRAVEL = {LONGEST_TRAVEL!r};
const float PACK_SMALLEST_DROP = {_SMALLEST_DROP!r};
const int PACK_HALVINGS = {HALVINGS};
const float PACK_SOLID_HEIGHT = {SOLID_HEIGHT!r};
const float PACK_LIT_HEIGHT = {LIT_HEIGHT!r};

vec2 packSightTravel(vec2 fromCamera, float drop) {{
    float slope = PACK_SLAB_HEIGHT / max(drop, PACK_SMALLEST_DROP);
    float held = min(PACK_ALL, PACK_LONGEST_TRAVEL / max(length(fromCamera * slope), PACK_SMALLEST_DROP));
    return fromCamera * slope * held;
}}

float packRayHeight(bool rising, float progress) {{
    return rising ? progress : PACK_ALL - progress;
}}

bool packIsUnder(ivec2 block, vec2 entry, vec2 travel, bool rising, float progress, float kind, float clock,
        float footprint) {{
    return packRayHeight(rising, progress)
        < packLayerHeight(block, entry + travel * progress, kind, clock, footprint);
}}

float packLookedInto(vec2 column) {{
    float deep = smoothstep(PACK_NONE, PACK_SOLID_HEIGHT, column.y);
    return max(column.x, PACK_FULL_CLOUD - (PACK_FULL_CLOUD - column.x) * (PACK_ALL - deep));
}}

// What is seen is a vec2: the height it is met at, and how much cloud is looked into there.
vec2 packSeenCloud(ivec2 block, vec2 entry, vec2 travel, bool rising, int steps, float clock, float footprint) {{
    float kind = packPlaceKind(block, entry);
    vec2 column = packLayerColumn(block, entry, kind, clock, footprint);
    if (steps == PACK_NO_STEPS) {{
        return vec2(rising ? PACK_NONE : column.y, packLookedInto(column));
    }}
    float base = rising ? packLookedInto(column) : PACK_NO_CLOUD;
    if (base >= PACK_FULL_CLOUD) {{
        return vec2(PACK_NONE, base);
    }}
    float highest = packTallestTop(kind);
    float clearUntil = PACK_NONE;
    for (int step = 1; step <= steps; step++) {{
        float underFrom = float(step) / float(steps);
        if (packRayHeight(rising, underFrom) < highest
                && packIsUnder(block, entry, travel, rising, underFrom, kind, clock, footprint)) {{
            for (int halving = 0; halving < PACK_HALVINGS; halving++) {{
                float middle = (clearUntil + underFrom) * PACK_HALF;
                if (packIsUnder(block, entry, travel, rising, middle, kind, clock, footprint)) {{
                    underFrom = middle;
                }} else {{
                    clearUntil = middle;
                }}
            }}
            float further = min(underFrom + PACK_ALL / float(steps), PACK_ALL);
            vec2 met = packLayerColumn(block, entry + travel * underFrom, kind, clock, footprint);
            vec2 beyond = packLayerColumn(block, entry + travel * further, kind, clock, footprint);
            return vec2(
                packRayHeight(rising, underFrom), max(base, max(packLookedInto(met), packLookedInto(beyond))));
        }}
        clearUntil = underFrom;
    }}
    if (!rising) {{
        return vec2(PACK_NONE, packLookedInto(packLayerColumn(block, entry + travel, kind, clock, footprint)));
    }}
    return vec2(PACK_NONE, base);
}}

float packCloudAlpha(float base, float density) {{
    return PACK_OPAQUE - pow(PACK_OPAQUE - min(base, PACK_DENSEST_ALPHA), density);
}}

float packLift(float height, float density) {{
    float lit = smoothstep(PACK_NONE, PACK_LIT_HEIGHT, height);
    return PACK_ALL - density * (PACK_ALL - lit);
}}

float packLightGain(float shade, float lit) {{
    return shade * (PACK_ALL - lit) + PACK_NO_GAIN * lit;
}}
""")
