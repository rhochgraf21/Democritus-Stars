"""The forms of cloud: how much cloud stands over each place of the layer, and how high its top is.

A form is a set of numbers (Shape) for a sum of noises: several lattices, each half the size of the one
before and weaker, the coarse ones giving a cloud its outline and the fine ones its detail, moved up and
down by the smooth noise of a field that gathers the clouds in some places and clears others. Cirrus is
smooth noise on a lattice sixteen times longer than wide, thin and soft. Altocumulus is small billows,
which are its cloudlets. Cumulus is larger billows gathered into heaps. Stratus is smooth, very large and
mostly cloud.

Which form where: one more smooth noise, on lattices of its own that do not drift, gives every place a
kind from 0 to 1. The forms sit along that range in the order thin to heavy, and a place has the form its
kind is nearest to. Between two forms the one fades into the other over a short stretch.

Drift: the forms' noise is read DRIFT_SPAN blocks further east while the clock runs from 0 to 1, and
repeats every DRIFT_SPAN blocks from east to west, so that nothing jumps when the clock starts again.

Far away, each lattice is faded out as its corners near NARROWEST_PIXELS pixels and left at its middle
level below that, which keeps the amount of cloud and drops only the detail that would flicker.

The top of a cloud is a dome with steep sides, shaped by the coarsest lattices alone (BODY_OCTAVES).

The maths lives twice, as in shader.effects.clouds.noise.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import NewType

import deal

from shader.effects import GlslCode
from shader.effects.clouds.noise import (
    ALL,
    BILLOW_MEAN,
    HALF,
    HIGHEST,
    LARGEST_LATTICE,
    LOWEST,
    MIDDLE,
    NO_WRAP,
    NONE,
    BlockSpan,
    Lattice,
    LayerPlace,
    Level,
    Seed,
    Share,
    SizeExponent,
    WrapMask,
    billow_noise,
    is_lattice,
    is_level,
    is_placed,
    smoothstep,
    value_noise,
)

# How many blocks of the cloud layer one pixel of the screen covers.
Footprint = NewType("Footprint", float)

# How much cloud there is at a place: 0 is clear sky, 1 is the thickest cloud.
Density = NewType("Density", float)

# What a colour is multiplied by.
Gain = NewType("Gain", float)

# The game's clock, GameTime of the Globals block: 0 to 1 over 24000 ticks, then 0 again.
Clock = NewType("Clock", float)

# How many lattices a form adds up.
OctaveCount = NewType("OctaveCount", int)


class Grain(Enum):
    """What the detail of a form is made of."""

    SMOOTH = "smooth"
    BILLOWS = "billows"


class Form(Enum):
    """A form of cloud, in the order thin to heavy, which is the order the kinds of place run in."""

    CIRRUS = "cirrus"
    ALTOCUMULUS = "altocumulus"
    CUMULUS = "cumulus"
    STRATUS = "stratus"


@dataclass(frozen=True)
class Shape:
    """The numbers that make the noise of one form.

    The detail is made of `grain` on `lattice` and on `octaves` lattices in all, each half the size of
    the one before and `gain` times as strong. The smooth noise of the `field` lattice moves the level
    of the detail up and down by `field_swing`, which gathers the clouds in some places and clears
    others. Under the level `clear_below` there is no cloud, and over the next `edge` the cloud grows
    to `thickness`. Its top rises to `height`, a share of the slab.
    """

    field: Lattice
    field_swing: Level
    grain: Grain
    lattice: Lattice
    octaves: OctaveCount
    gain: Gain
    clear_below: Level
    edge: Level
    thickness: Density
    height: Share
    seed: Seed


@dataclass(frozen=True)
class Column:
    """The cloud over one place: how much there is, and how high its top stands in the slab, from 0 to 1."""

    density: Density
    height: Share

# Clear sky.
NO_CLOUD = Density(0.0)

# As much cloud as cloud gets.
FULL_CLOUD = Density(1.0)

# The gain that leaves a colour as it is.
NO_GAIN = Gain(1.0)

# The clock where it starts.
CLOCK_START = Clock(0.0)

# The clock where it ends, just before it starts again.
CLOCK_END = Clock(1.0)

# From this footprint on, a pixel covers so much of the layer that the height of a cloud cannot be seen
# in it, and the cloud is drawn flat.
FLAT_FOOTPRINT = Footprint(2.0)

# How far the clouds move while the clock runs once, as a power of two of blocks: 2048 blocks in 20
# minutes, 1.7 blocks a second. The game's own clouds move 0.6. Faster, and the drifting noise can repeat
# less often: it repeats once every span, and at 2048 blocks a copy is beyond what a player takes in at once.
DRIFT_EXPONENT = LARGEST_LATTICE

DRIFT_SPAN = BlockSpan(float(1 << DRIFT_EXPONENT))

# The top of a cloud goes on rising over this many of its edges. At 1 a top would be level wherever the
# cloud is full, and every cumulus a table.
TOWER = 1.5

# The top of a cloud is shaped by this many of a form's lattices, coarsest first. One gives every cloud the
# same dome, and two still give lumps too even to pass for cloud; more would cost more at every look for
# bumps too small to make out.
BODY_OCTAVES = OctaveCount(3)

# A lattice is left out when its corners are this many pixels apart or fewer, and is whole at twice that.
NARROWEST_PIXELS = 2.0

# Keeps the footprint of a pixel from being nothing, which no pixel's is, where it is divided by.
_SMALLEST_FOOTPRINT = Footprint(1e-6)

# The kinds of place run over a lattice of 2048 blocks and two finer ones, which bend the borders between
# forms away from the lines of the lattice: a player sees more than one form in a sky and has to travel to
# stand under another.
REGION_LATTICE = Lattice(SizeExponent(11), SizeExponent(11))

REGION_OCTAVES = OctaveCount(3)

REGION_GAIN = Gain(0.5)

# Added-up noise gathers about its middle, which would give nearly every place one of the two middle
# forms. The kind is spread about the middle by this much, so the forms at the ends get their share.
REGION_CONTRAST = 2.5

# How much faster than its kind changes one form gives way to the next. At 1 two forms would lie over each
# other for hundreds of blocks, which reads as neither.
REGION_SHARPNESS = 4.0

# A form's field takes the seed after those of its lattices.
_FIELD_SEED_OFFSET = 8

# Seeds only have to differ. A form's lattices take the seeds from its own onward, so forms are this far apart.
_SEED_STRIDE = 16

MOST_OCTAVES = OctaveCount(5)

_SHAPES = {
    # Streaks along the drift, 1024 blocks long and 64 wide, thin enough to see the sky through.
    Form.CIRRUS: Shape(
        Lattice(SizeExponent(10), SizeExponent(8)), Level(0.5), Grain.SMOOTH,
        Lattice(SizeExponent(10), SizeExponent(6)), OctaveCount(4), Gain(0.7), Level(0.55), Level(0.25),
        Density(0.4), Share(0.1), Seed(0 * _SEED_STRIDE),
    ),
    # Fields of cloudlets some 40 blocks across.
    Form.ALTOCUMULUS: Shape(
        Lattice(SizeExponent(9), SizeExponent(9)), Level(0.5), Grain.BILLOWS,
        Lattice(SizeExponent(6), SizeExponent(6)), OctaveCount(2), Gain(0.5), Level(0.65), Level(0.25),
        Density(0.85), Share(0.45), Seed(1 * _SEED_STRIDE),
    ),
    # Heaps of billows with sky between the heaps.
    Form.CUMULUS: Shape(
        Lattice(SizeExponent(8), SizeExponent(8)), Level(0.7), Grain.BILLOWS,
        Lattice(SizeExponent(7), SizeExponent(7)), OctaveCount(3), Gain(0.6), Level(0.7), Level(0.3),
        Density(1.0), Share(1.0), Seed(2 * _SEED_STRIDE),
    ),
    # A sheet with few breaks, uneven in thickness.
    Form.STRATUS: Shape(
        Lattice(SizeExponent(10), SizeExponent(10)), Level(0.3), Grain.SMOOTH,
        Lattice(SizeExponent(9), SizeExponent(9)), OctaveCount(4), Gain(0.5), Level(0.3), Level(0.3),
        Density(0.9), Share(0.5), Seed(3 * _SEED_STRIDE),
    ),
}

_REGION_SEED = Seed(len(Form) * _SEED_STRIDE)

_FORMS = tuple(Form)

_LAST_FORM = len(_FORMS) - 1


@deal.pure
@deal.post(lambda result: is_lattice(result.lattice) and 0 < result.octaves <= MOST_OCTAVES)
@deal.post(lambda result: min(result.lattice.east, result.lattice.south) - (result.octaves - 1) >= 0)
@deal.post(lambda result: is_lattice(result.field) and result.octaves <= _FIELD_SEED_OFFSET < _SEED_STRIDE)
@deal.post(lambda result: is_level(result.clear_below) and result.edge > LOWEST)
@deal.post(lambda result: NO_CLOUD < result.thickness <= FULL_CLOUD and NONE < result.height <= ALL)
def shape_of(form: Form) -> Shape:
    """The numbers that make the noise of a form."""
    return _SHAPES[form]


@deal.pure
@deal.pre(lambda lattice: is_lattice(lattice))
@deal.post(lambda result: result >= 0 and result & (result + 1) == 0)
def drift_wrap(lattice: Lattice) -> WrapMask:
    """The wrap of a lattice that drifts: it repeats once every DRIFT_SPAN blocks from east to west."""
    return WrapMask((1 << (DRIFT_EXPONENT - lattice.east)) - 1)


@deal.pure
@deal.pre(lambda corners_apart, footprint: corners_apart > 0 and footprint >= 0)
# Two lengths without end have no ratio: a lattice and a footprint are both lengths in the world.
@deal.pre(lambda corners_apart, footprint: math.isfinite(corners_apart) and math.isfinite(footprint))
@deal.post(lambda result: NONE <= result <= ALL)
@deal.ensure(
    lambda corners_apart, footprint, result: corners_apart > footprint * NARROWEST_PIXELS or result == NONE
)
def octave_keep(corners_apart: BlockSpan, footprint: Footprint) -> Share:
    """How much of a lattice is drawn: none of it once its corners are NARROWEST_PIXELS pixels apart or fewer."""
    pixels = corners_apart / max(footprint, _SMALLEST_FOOTPRINT)
    return Share(min(max(pixels / NARROWEST_PIXELS - ALL, NONE), ALL))


@deal.pure
@deal.post(is_level)
def resting_level(grain: Grain) -> Level:
    """The level a lattice of a grain is left at when it is too far away to draw: the average of its noise."""
    return BILLOW_MEAN if grain is Grain.BILLOWS else MIDDLE


def _detail(shape: Shape, place: LayerPlace, octave: int, footprint: Footprint) -> Level:
    """One lattice of a form's detail at a place, drawn as far as the footprint lets it be."""
    lattice = Lattice(SizeExponent(shape.lattice.east - octave), SizeExponent(shape.lattice.south - octave))
    keep = octave_keep(BlockSpan(float(1 << min(lattice.east, lattice.south))), footprint)
    resting = resting_level(shape.grain)
    noise = resting
    if keep > NONE:
        make = billow_noise if shape.grain is Grain.BILLOWS else value_noise
        noise = make(place, lattice, drift_wrap(lattice), Seed(shape.seed + octave))
    return Level(resting + keep * (noise - resting))


def _field(shape: Shape, place: LayerPlace) -> float:
    """How far the field of a form moves the level of its detail at a place."""
    field = value_noise(place, shape.field, drift_wrap(shape.field), Seed(shape.seed + _FIELD_SEED_OFFSET))
    return (field - MIDDLE) * shape.field_swing


@deal.pure
@deal.pre(lambda footprint: footprint >= 0)
@deal.post(lambda result: NONE <= result <= ALL)
@deal.ensure(lambda footprint, result: footprint < FLAT_FOOTPRINT or result == NONE)
def relief(footprint: Footprint) -> Share:
    """How much of its height a cloud keeps for a pixel of a footprint: all of it near, none from FLAT_FOOTPRINT on.

    It goes over the second half of the way there, so that a cloud sinks flat as it draws away and is not
    tall in one frame and flat in the next.
    """
    return Share(ALL - smoothstep(Level(FLAT_FOOTPRINT * HALF), Level(FLAT_FOOTPRINT), Level(footprint)))


def _body(shape: Shape, level: float, footprint: Footprint) -> Column:
    """The body of a cloud where its coarsest lattices and its field give this level: a dome with steep sides.

    The dome leaves its rim steeply and is level at its top, 1 - (1 - rise) squared. A curve with a crease
    in it, at the rim or where the top is reached, shows on every cloud as a line around it.
    """
    rise = min(max((level - shape.clear_below) / (shape.edge * TOWER), LOWEST), HIGHEST)
    return Column(
        Density(smoothstep(shape.clear_below, shape.clear_below + shape.edge, level) * shape.thickness),
        Share((ALL - (ALL - rise) * (ALL - rise)) * shape.height * relief(footprint)),
    )


@deal.pure
@deal.pre(lambda form, place, footprint: is_placed(place) and footprint >= 0)
@deal.post(lambda result: NONE <= result <= ALL)
@deal.ensure(lambda form, place, footprint, result: result <= shape_of(form).height)
def form_height(form: Form, place: LayerPlace, footprint: Footprint) -> Share:
    """How high the top of a form's cloud stands over a place of its drifting noise, as a share of the slab.

    The height comes from the first BODY_OCTAVES lattices and the field alone: it is what a line of sight
    is tested against at every look.
    """
    shape = shape_of(form)
    total = 0.0
    weight = 0.0
    amplitude = NO_GAIN
    for octave in range(min(shape.octaves, BODY_OCTAVES)):
        total += amplitude * _detail(shape, place, octave, footprint)
        weight += amplitude
        amplitude = Gain(amplitude * shape.gain)
    return _body(shape, total / weight + _field(shape, place), footprint).height


@deal.pure
@deal.pre(lambda form, place, footprint: is_placed(place) and footprint >= 0)
@deal.post(lambda result: NO_CLOUD <= result.density <= FULL_CLOUD and NONE <= result.height <= ALL)
@deal.ensure(lambda form, place, footprint, result: result.density <= shape_of(form).thickness)
@deal.ensure(lambda form, place, footprint, result: result.height == form_height(form, place, footprint))
@deal.ensure(lambda form, place, footprint, result: result.height == NONE or result.density > NO_CLOUD)
def form_column(form: Form, place: LayerPlace, footprint: Footprint) -> Column:
    """The cloud a form has over a place of its drifting noise, for a pixel that covers `footprint`.

    There is cloud wherever the body is, so that a top always has cloud under it, and wherever all the
    lattices together give some, which adds the wisps and the ragged edge.
    """
    shape = shape_of(form)
    total = 0.0
    weight = 0.0
    amplitude = NO_GAIN
    body_level = 0.0
    for octave in range(shape.octaves):
        total += amplitude * _detail(shape, place, octave, footprint)
        weight += amplitude
        amplitude = Gain(amplitude * shape.gain)
        if octave + 1 == min(shape.octaves, BODY_OCTAVES):
            body_level = total / weight
    moved = _field(shape, place)
    body = _body(shape, body_level + moved, footprint)
    detailed = smoothstep(shape.clear_below, shape.clear_below + shape.edge, total / weight + moved) * shape.thickness
    return Column(Density(max(body.density, detailed)), body.height)


@deal.pure
@deal.pre(lambda form, place, footprint: is_placed(place) and footprint >= 0)
@deal.post(lambda result: NO_CLOUD <= result <= FULL_CLOUD)
def form_density(form: Form, place: LayerPlace, footprint: Footprint) -> Density:
    """How much cloud a form has at a place of its drifting noise."""
    return form_column(form, place, footprint).density


@deal.pure
@deal.pre(lambda place: is_placed(place))
@deal.post(is_level)
def place_kind(place: LayerPlace) -> Level:
    """The kind of a place of the world, which says what form its clouds have: 0 is the thinnest, 1 the heaviest."""
    total = 0.0
    weight = 0.0
    amplitude = NO_GAIN
    for octave in range(REGION_OCTAVES):
        lattice = Lattice(SizeExponent(REGION_LATTICE.east - octave), SizeExponent(REGION_LATTICE.south - octave))
        total += amplitude * value_noise(place, lattice, NO_WRAP, Seed(_REGION_SEED + octave))
        weight += amplitude
        amplitude = Gain(amplitude * REGION_GAIN)
    return Level(min(max((total / weight - MIDDLE) * REGION_CONTRAST + MIDDLE, LOWEST), HIGHEST))


@deal.pure
@deal.pre(lambda kind, form: is_level(kind))
@deal.post(lambda result: NONE <= result <= ALL)
def form_share(kind: Level, form: Form) -> Share:
    """How much of the cloud at a place of this kind is of a form: all of it where the kind is the form's own."""
    own_kind = _FORMS.index(form) / _LAST_FORM
    nearness = ALL - abs(kind - own_kind) * _LAST_FORM
    return Share(min(max((nearness - HALF) * REGION_SHARPNESS + HALF, NONE), ALL))


@deal.pure
@deal.pre(lambda place, clock: is_placed(place) and CLOCK_START <= clock <= CLOCK_END)
@deal.ensure(lambda place, clock, result: result.east_rest >= place.east_rest)
@deal.ensure(lambda place, clock, result: (result.east_block, result.south_block) == (place.east_block, place.south_block))
@deal.ensure(lambda place, clock, result: result.south_rest == place.south_rest)
def drifted(place: LayerPlace, clock: Clock) -> LayerPlace:
    """The place of the drifting noise that lies over a place of the world at a moment of the clock.

    The noise is read further east as the clock runs, so what is drawn moves west.
    """
    return LayerPlace(
        place.east_block, place.south_block, BlockSpan(place.east_rest + clock * DRIFT_SPAN), place.south_rest
    )


@deal.pure
@deal.pre(lambda place, kind, clock, footprint: is_placed(place) and CLOCK_START <= clock <= CLOCK_END)
@deal.pre(lambda place, kind, clock, footprint: footprint >= 0 and is_level(kind))
@deal.post(lambda result: NONE <= result <= ALL)
def layer_height(place: LayerPlace, kind: Level, clock: Clock, footprint: Footprint) -> Share:
    """How high the top of the cloud stands over a place whose clouds are of a kind, at a moment of the clock."""
    moved = drifted(place, clock)
    height = 0.0
    for form in _FORMS:
        share = form_share(kind, form)
        if share > NONE:
            height += share * form_height(form, moved, footprint)
    return Share(min(height, ALL))


@deal.pure
@deal.pre(lambda place, kind, clock, footprint: is_placed(place) and CLOCK_START <= clock <= CLOCK_END)
@deal.pre(lambda place, kind, clock, footprint: footprint >= 0 and is_level(kind))
@deal.post(lambda result: NO_CLOUD <= result.density <= FULL_CLOUD and NONE <= result.height <= ALL)
def layer_column(place: LayerPlace, kind: Level, clock: Clock, footprint: Footprint) -> Column:
    """The cloud over a place whose clouds are of a kind, at a moment of the clock."""
    moved = drifted(place, clock)
    density = 0.0
    height = 0.0
    for form in _FORMS:
        share = form_share(kind, form)
        if share > NONE:
            column = form_column(form, moved, footprint)
            density += share * column.density
            height += share * column.height
    return Column(Density(min(density, FULL_CLOUD)), Share(min(height, ALL)))


@deal.pure
@deal.pre(lambda place, clock, footprint: is_placed(place) and CLOCK_START <= clock <= CLOCK_END)
@deal.pre(lambda place, clock, footprint: footprint >= 0)
@deal.post(lambda result: NO_CLOUD <= result <= FULL_CLOUD)
def layer_density(place: LayerPlace, clock: Clock, footprint: Footprint) -> Density:
    """How much cloud there is at a place of the world at a moment of the clock."""
    return layer_column(place, place_kind(place), clock, footprint).density


@deal.pure
@deal.pre(lambda kind: is_level(kind))
@deal.post(lambda result: NONE <= result <= ALL)
def tallest_top(kind: Level) -> Share:
    """The highest a top can stand over a place of a kind, as a share of the slab.

    A line of sight that is higher than this is under no top, which is known without a look.
    """
    return Share(min(math.fsum(form_share(kind, form) * shape_of(form).height for form in _FORMS), ALL))

_GLSL_GRAINS = {Grain.SMOOTH: "PACK_GRAIN_SMOOTH", Grain.BILLOWS: "PACK_GRAIN_BILLOWS"}


def _glsl_shape(form: Form) -> str:
    shape = shape_of(form)
    return (
        f"    PackShape(ivec2({shape.field.east}, {shape.field.south}), {shape.field_swing!r},"
        f" {_GLSL_GRAINS[shape.grain]}, ivec2({shape.lattice.east}, {shape.lattice.south}), {shape.octaves},"
        f" {shape.gain!r}, {shape.clear_below!r}, {shape.edge!r}, {shape.thickness!r}, {shape.height!r},"
        f" {shape.seed})"
    )

_SHAPE_LINES = ",\n".join(_glsl_shape(form) for form in _FORMS)

# The forms in GLSL, with the constants they are written from, to go after the noise. Each function is the
# Python function of the same name, line for line. Change one and change the other.
GLSL = GlslCode(f"""\
const int PACK_GRAIN_SMOOTH = 0;
const int PACK_GRAIN_BILLOWS = 1;

struct PackShape {{
    ivec2 field;
    float fieldSwing;
    int grain;
    ivec2 lattice;
    int octaves;
    float gain;
    float clearBelow;
    float edge;
    float thickness;
    float height;
    int seed;
}};

const int PACK_FORMS = {len(_FORMS)};
const int PACK_LAST_FORM = PACK_FORMS - 1;
const PackShape PACK_SHAPES[PACK_FORMS] = PackShape[PACK_FORMS](
{_SHAPE_LINES}
);

const float PACK_NO_CLOUD = {NO_CLOUD!r};
const float PACK_FULL_CLOUD = {FULL_CLOUD!r};
const float PACK_NO_GAIN = {NO_GAIN!r};
const float PACK_TOWER = {TOWER!r};
const float PACK_FLAT_FOOTPRINT = {FLAT_FOOTPRINT!r};
const int PACK_DRIFT_EXPONENT = {DRIFT_EXPONENT};
const float PACK_DRIFT_SPAN = {DRIFT_SPAN!r};
const float PACK_NARROWEST_PIXELS = {NARROWEST_PIXELS!r};
const float PACK_SMALLEST_FOOTPRINT = {_SMALLEST_FOOTPRINT!r};
const ivec2 PACK_REGION_LATTICE = ivec2({REGION_LATTICE.east}, {REGION_LATTICE.south});
const int PACK_REGION_OCTAVES = {REGION_OCTAVES};
const float PACK_REGION_GAIN = {REGION_GAIN!r};
const float PACK_REGION_CONTRAST = {REGION_CONTRAST!r};
const float PACK_REGION_SHARPNESS = {REGION_SHARPNESS!r};
const int PACK_REGION_SEED = {_REGION_SEED};
const int PACK_BODY_OCTAVES = {BODY_OCTAVES};
const int PACK_FIELD_SEED_OFFSET = {_FIELD_SEED_OFFSET};

int packDriftWrap(ivec2 lattice) {{
    return (1 << (PACK_DRIFT_EXPONENT - lattice.x)) - 1;
}}

float packOctaveKeep(float cornersApart, float footprint) {{
    float pixels = cornersApart / max(footprint, PACK_SMALLEST_FOOTPRINT);
    return clamp(pixels / PACK_NARROWEST_PIXELS - PACK_ALL, PACK_NONE, PACK_ALL);
}}

float packRestingLevel(int grain) {{
    return grain == PACK_GRAIN_BILLOWS ? PACK_BILLOW_MEAN : PACK_MIDDLE;
}}

float packDetail(PackShape shape, ivec2 block, vec2 rest, int octave, float footprint) {{
    ivec2 lattice = shape.lattice - octave;
    float keep = packOctaveKeep(float(1 << min(lattice.x, lattice.y)), footprint);
    float resting = packRestingLevel(shape.grain);
    float noise = resting;
    if (keep > PACK_NONE) {{
        noise = shape.grain == PACK_GRAIN_BILLOWS
            ? packBillowNoise(block, rest, lattice, packDriftWrap(lattice), shape.seed + octave)
            : packValueNoise(block, rest, lattice, packDriftWrap(lattice), shape.seed + octave);
    }}
    return resting + keep * (noise - resting);
}}

float packField(PackShape shape, ivec2 block, vec2 rest) {{
    float field = packValueNoise(
        block, rest, shape.field, packDriftWrap(shape.field), shape.seed + PACK_FIELD_SEED_OFFSET);
    return (field - PACK_MIDDLE) * shape.fieldSwing;
}}

// A column is a vec2: how much cloud there is over a place, and how high its top stands in the slab.
float packRelief(float footprint) {{
    return PACK_ALL - smoothstep(PACK_FLAT_FOOTPRINT * PACK_HALF, PACK_FLAT_FOOTPRINT, footprint);
}}

vec2 packBody(PackShape shape, float level, float footprint) {{
    float rise = clamp((level - shape.clearBelow) / (shape.edge * PACK_TOWER), PACK_LOWEST, PACK_HIGHEST);
    return vec2(
        smoothstep(shape.clearBelow, shape.clearBelow + shape.edge, level) * shape.thickness,
        (PACK_ALL - (PACK_ALL - rise) * (PACK_ALL - rise)) * shape.height * packRelief(footprint));
}}

float packFormHeight(int form, ivec2 block, vec2 rest, float footprint) {{
    PackShape shape = PACK_SHAPES[form];
    float total = 0.0;
    float weight = 0.0;
    float amplitude = PACK_NO_GAIN;
    for (int octave = 0; octave < min(shape.octaves, PACK_BODY_OCTAVES); octave++) {{
        total += amplitude * packDetail(shape, block, rest, octave, footprint);
        weight += amplitude;
        amplitude *= shape.gain;
    }}
    return packBody(shape, total / weight + packField(shape, block, rest), footprint).y;
}}

vec2 packFormColumn(int form, ivec2 block, vec2 rest, float footprint) {{
    PackShape shape = PACK_SHAPES[form];
    float total = 0.0;
    float weight = 0.0;
    float amplitude = PACK_NO_GAIN;
    float bodyLevel = 0.0;
    for (int octave = 0; octave < shape.octaves; octave++) {{
        total += amplitude * packDetail(shape, block, rest, octave, footprint);
        weight += amplitude;
        amplitude *= shape.gain;
        if (octave + 1 == min(shape.octaves, PACK_BODY_OCTAVES)) {{
            bodyLevel = total / weight;
        }}
    }}
    float moved = packField(shape, block, rest);
    vec2 body = packBody(shape, bodyLevel + moved, footprint);
    float detailed = smoothstep(shape.clearBelow, shape.clearBelow + shape.edge, total / weight + moved)
        * shape.thickness;
    return vec2(max(body.x, detailed), body.y);
}}

float packPlaceKind(ivec2 block, vec2 rest) {{
    float total = 0.0;
    float weight = 0.0;
    float amplitude = PACK_NO_GAIN;
    for (int octave = 0; octave < PACK_REGION_OCTAVES; octave++) {{
        total += amplitude * packValueNoise(
            block, rest, PACK_REGION_LATTICE - octave, PACK_NO_WRAP, PACK_REGION_SEED + octave);
        weight += amplitude;
        amplitude *= PACK_REGION_GAIN;
    }}
    return clamp((total / weight - PACK_MIDDLE) * PACK_REGION_CONTRAST + PACK_MIDDLE, PACK_LOWEST, PACK_HIGHEST);
}}

float packFormShare(float kind, int form) {{
    float ownKind = float(form) / float(PACK_LAST_FORM);
    float nearness = PACK_ALL - abs(kind - ownKind) * float(PACK_LAST_FORM);
    return clamp((nearness - PACK_HALF) * PACK_REGION_SHARPNESS + PACK_HALF, PACK_NONE, PACK_ALL);
}}

float packLayerHeight(ivec2 block, vec2 rest, float kind, float clock, float footprint) {{
    vec2 moved = vec2(rest.x + clock * PACK_DRIFT_SPAN, rest.y);
    float height = PACK_NONE;
    for (int form = 0; form < PACK_FORMS; form++) {{
        float share = packFormShare(kind, form);
        if (share > PACK_NONE) {{
            height += share * packFormHeight(form, block, moved, footprint);
        }}
    }}
    return min(height, PACK_ALL);
}}

vec2 packLayerColumn(ivec2 block, vec2 rest, float kind, float clock, float footprint) {{
    vec2 moved = vec2(rest.x + clock * PACK_DRIFT_SPAN, rest.y);
    vec2 column = vec2(PACK_NO_CLOUD, PACK_NONE);
    for (int form = 0; form < PACK_FORMS; form++) {{
        float share = packFormShare(kind, form);
        if (share > PACK_NONE) {{
            column += share * packFormColumn(form, block, moved, footprint);
        }}
    }}
    return min(column, vec2(PACK_FULL_CLOUD, PACK_ALL));
}}

// The shape is copied out of the array before its height is read. Reading one member straight out of the
// constant array, PACK_SHAPES[form].height, crashes Apple's shader compiler on an M1 ("unable to legalize
// instruction"), and the game then hangs on the first frame that draws a cloud.
float packTallestTop(float kind) {{
    float highest = PACK_NONE;
    for (int form = 0; form < PACK_FORMS; form++) {{
        PackShape shape = PACK_SHAPES[form];
        highest += packFormShare(kind, form) * shape.height;
    }}
    return min(highest, PACK_ALL);
}}
""")
