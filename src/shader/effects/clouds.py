"""Clouds as soft puffs: each cloud cell of the game drawn as a puff that fades to nothing at its edge.

The game keeps its clouds as a grid of cells, 12 blocks wide and 4 high, and draws each cell as a flat
box: clouds.vsh builds every face of every cell from the CloudFaces buffer, and clouds.fsh fills it with
one colour. The cells are kept, because they are what drifts, sits at a real height, hides behind
terrain and takes the colour of the weather. Only how a cell is drawn changes.

A puff is the top or the bottom face of a cell, grown about the centre of the cell until it reaches the
centres of the cells beside it. Its density is 1 at its centre and falls to 0 at its edge. The side faces
of a cell are not drawn: a wall has a hard edge, and nothing in a face says how far the cloud goes on
behind it.

Why the falloff has this shape. A puff covers its neighbours, so a cloud of many cells is many puffs on
top of each other, and a falloff chosen by eye leaves a grid of bright dots or of dark seams. This one
is chosen so that the grid cannot show: along each axis the falloff of a puff and the falloff of the
puff one cell on add up to exactly 1 at every point between their centres. Wherever the four cells
around a point are all cloud, the densities of their puffs add up to 1, the same as at the centre of a
single puff. The edge of a cloud is where some of the four are missing, and there the sum falls smoothly
to 0.

Densities add, alphas do not: two layers of alpha 0.4 hide less than one of 0.8. So a puff's alpha is
what a slab of cloud of the puff's density would hide, 1 - (1 - base) ** density, and alpha blending of
the puffs then multiplies out to exactly the base alpha wherever the densities add up to 1.

What the game sends (CloudRenderer and RenderPipelines of 26.3), and what is made of it:

- Fancy clouds send the top face of a cell unless the camera is below the layer, the bottom face unless
  it is above, a side face where the neighbour on that side is empty and the face looks toward the
  camera, and all six faces again, marked as inside faces, for the nine cells around the camera. The
  neighbours the game knows are the four that share an edge; cells that meet at a corner are not
  neighbours. Fast clouds send one bottom face per cell, marked to take the top colour.
- From below a pixel sees the puffs of the bottom faces, from above those of the top faces. Between the
  two, where the game sends both and culls both for facing away, each is turned to face the camera, so
  the layer does not vanish while the player flies through it. Inside faces would draw the same puffs a
  second time and are not drawn.
- The game cannot be asked which cell of the world a face belongs to: the cell numbers count from the
  camera's cell and change when the camera crosses a cell. A noise by cell would jump each time, and
  there is nothing else to anchor one to, so the puffs carry none.

Depth. Every way the game draws clouds but one writes depth (the plain pipelines always, the fancy
order-independent one in its depth bounds phase), and tests later fragments against it. Vanilla's faces
never overlap, so that costs nothing there. Puffs of one layer overlap in one plane, where two triangles
give depths that differ in the last bits, and whichever loses is not drawn. So every puff writes its
depth with the last bits cleared: fragments of one plane at one pixel then hold the same depth and all
pass. The colour-less phases of order-independent transparency get the same alpha as the colour phase,
so the transmittance they store is the one the colour is drawn with.

Cost: a pixel is covered by at most four puffs, those of the four cells around it, where vanilla draws
one face. Each costs one pow and a few multiplications, with no loop and no texture read.

The maths lives twice: as the functions below, which the tests check, and as the GLSL that `effect`
attaches, which the game runs. GLSL cannot be run where the tests run, and Python cannot be run in the
game. Both are written from the same named constants, and each GLSL function mirrors the Python one of
the same name line for line, so that the two can be compared by eye.

Not yet seen in the game: that puffs of one plane all pass the depth test once their depths share their
last bits, and how the layer looks from inside it.
"""

import math
from dataclasses import dataclass
from typing import NewType

import deal

from shader.effects import Attachment, Effect, EffectName, GlslCode, Place, Position
from shader.vanilla import ShaderPath

# A length across the cloud layer, counted in cloud cells. The game makes a cell 12 blocks wide.
CellSpan = NewType("CellSpan", float)

# How far a point is from the centre of a puff along one axis: 0 at the centre, 1 or -1 at the edge.
Reach = NewType("Reach", float)

# How much cloud there is at a point, as a share of the cloud at the centre of a single puff.
Density = NewType("Density", float)

# How much of what lies behind a fragment it hides: 0 for none, 1 for all.
Alpha = NewType("Alpha", float)

# How many of the lowest bits of a depth are cleared.
BitCount = NewType("BitCount", int)


@dataclass(frozen=True)
class LayerPoint:
    """A point of the cloud layer seen from above, in cells. The centre of a cell is a point too."""

    east: CellSpan
    south: CellSpan


@dataclass(frozen=True)
class PuffPosition:
    """Where a point lies in one puff: how far from the centre toward the edge, along each axis."""

    across: Reach
    along: Reach


NAME = EffectName("clouds")
VERTEX_SHADER = ShaderPath("core/clouds.vsh")
FRAGMENT_SHADER = ShaderPath("core/clouds.fsh")

# The lines of the game's clouds.vsh and clouds.fsh that the code attaches to.
VERTEX_COLOUR_OUTPUT = Place("layout(location = 1) out vec4 vertexColor;")
VERTEX_POSITION = Place("vec3 pos = (faceVertex * CellSize) + (vec3(cellX, 0, cellZ) * CellSize) + CloudOffset;")
VERTEX_COLOUR = Place("vertexColor = (useTopColor ? faceColors[1] : faceColors[direction]) * CloudColor;")
FRAGMENT_COLOUR_INPUT = Place("layout(location = 1) in vec4 vertexColor;")
FRAGMENT_PHASES = Place("#ifdef OIT_ALPHA_ONLY")

# From the centre of a puff to its edge. One cell is the only size at which the falloffs of two puffs one
# cell apart add up to 1 between their centres: smaller and a cloud has seams, larger and it has ridges.
PUFF_HALF_SIZE = CellSpan(1.0)

# Where the centre of a cell lies between its two edges. A face of the game runs from 0 to 1.
CELL_CENTRE = CellSpan(0.5)

CENTRE = Reach(0.0)
EDGE = Reach(1.0)

NO_CLOUD = Density(0.0)
FULL_CLOUD = Density(1.0)

CLEAR = Alpha(0.0)
OPAQUE = Alpha(1.0)

# The most a puff hides at its centre. The game's clouds are 0.8. A wholly opaque cloud lets nothing
# through to share out between overlapping puffs: every density above none would hide everything, and a
# puff would be a hard square twice as wide as its cell. Held just under opaque, the edge stays soft.
DENSEST_ALPHA = Alpha(0.95)

# The two numbers of the smoothstep curve t * t * (3 - 2 * t). It is level at both ends, so a puff has no
# crease at its centre or its edge, and it turns into 1 minus itself when t turns into 1 - t, which is
# what makes two neighbouring falloffs add up to 1.
_SMOOTH_OFFSET = 3.0
_SMOOTH_SLOPE = 2.0

# How far 1 - (1 - alpha) may land from alpha: the two subtractions each round in the last bits.
_ROUNDING = 1e-12

# A depth is a 32 bit float, and this many of its bits are its fraction.
_DEPTH_FRACTION_BITS = BitCount(23)

# Clearing 16 leaves 128 steps for each doubling of
# the distance, so a cloud is at most 0.8 % further away than it is when terrain is tested against it,
# and two depths of one plane, a few last bits apart, fall on two sides of a step about once in ten
# thousand pixels.
SHARED_DEPTH_BITS = BitCount(16)


def _is_placed(point: LayerPoint) -> bool:
    return math.isfinite(point.east) and math.isfinite(point.south)


@deal.pure
@deal.pre(lambda reach: not math.isnan(reach))
@deal.post(lambda result: NO_CLOUD <= result <= FULL_CLOUD)
@deal.ensure(lambda reach, result: reach != CENTRE or result == FULL_CLOUD)
@deal.ensure(lambda reach, result: abs(reach) < EDGE or result == NO_CLOUD)
def falloff(reach: Reach) -> Density:
    """The density of a puff along one axis: 1 at its centre, falling to exactly 0 at its edge and beyond."""
    t = min(abs(reach), EDGE)
    return Density(FULL_CLOUD - t * t * (_SMOOTH_OFFSET - _SMOOTH_SLOPE * t))


@deal.pure
@deal.pre(lambda point, centre: _is_placed(point) and _is_placed(centre))
@deal.ensure(lambda point, centre, result: (result.across == CENTRE) == (point.east == centre.east))
@deal.ensure(lambda point, centre, result: (result.along == CENTRE) == (point.south == centre.south))
@deal.ensure(lambda point, centre, result: (result.across > CENTRE) == (point.east > centre.east))
@deal.ensure(lambda point, centre, result: (result.along > CENTRE) == (point.south > centre.south))
def puff_position(point: LayerPoint, centre: LayerPoint) -> PuffPosition:
    """Where a point of the layer lies in the puff of the cell with this centre.

    The vertex stage gives the corners of a puff the positions 1 and -1, and the game interpolates
    between them for each fragment; this is the same value, reached from the point.
    """
    return PuffPosition(
        Reach((point.east - centre.east) / PUFF_HALF_SIZE),
        Reach((point.south - centre.south) / PUFF_HALF_SIZE),
    )


@deal.pure
@deal.pre(lambda position: not (math.isnan(position.across) or math.isnan(position.along)))
@deal.post(lambda result: NO_CLOUD <= result <= FULL_CLOUD)
@deal.ensure(
    lambda position, result: max(abs(position.across), abs(position.along)) < EDGE or result == NO_CLOUD
)
@deal.ensure(lambda position, result: result <= min(falloff(position.across), falloff(position.along)))
def puff_density(position: PuffPosition) -> Density:
    """The density of a puff at a position in it: 1 at its centre, exactly 0 on its edge and beyond."""
    return Density(falloff(position.across) * falloff(position.along))


@deal.pure
@deal.pre(lambda base, density: CLEAR <= base <= OPAQUE)
@deal.pre(lambda base, density: NO_CLOUD <= density <= FULL_CLOUD)
@deal.post(lambda result: CLEAR <= result <= DENSEST_ALPHA)
@deal.ensure(lambda base, density, result: density > NO_CLOUD or result == CLEAR)
@deal.ensure(lambda base, density, result: density < FULL_CLOUD or abs(result - min(base, DENSEST_ALPHA)) <= _ROUNDING)
def puff_alpha(base: Alpha, density: Density) -> Alpha:
    """How much a puff hides where it has this density, when the game's cloud colour hides `base`.

    What a slab of cloud lets through is multiplied when slabs are stacked, so a share of the density
    is a power of what the whole lets through, not a share of the alpha.
    """
    return Alpha(OPAQUE - (OPAQUE - min(base, DENSEST_ALPHA)) ** density)


@deal.pure
@deal.pre(lambda point, centres: _is_placed(point) and all(_is_placed(centre) for centre in centres))
@deal.post(lambda result: result >= NO_CLOUD)
@deal.ensure(lambda point, centres, result: len(centres) > 0 or result == NO_CLOUD)
def cloud_density(point: LayerPoint, centres: tuple[LayerPoint, ...]) -> Density:
    """The density at a point of a cloud made of the cells with these centres: the sum of their puffs."""
    return Density(math.fsum(puff_density(puff_position(point, centre)) for centre in centres))


@deal.pure
@deal.pre(lambda base, point, centres: CLEAR <= base <= OPAQUE)
@deal.pre(lambda base, point, centres: _is_placed(point) and all(_is_placed(centre) for centre in centres))
@deal.post(lambda result: CLEAR <= result <= OPAQUE)
@deal.ensure(lambda base, point, centres, result: len(centres) > 0 or result == CLEAR)
def cloud_alpha(base: Alpha, point: LayerPoint, centres: tuple[LayerPoint, ...]) -> Alpha:
    """How much a cloud of these cells hides at a point, once the game has blended its puffs.

    Each puff lets through what it does not hide, and what is let through by all of them is the
    product, in whatever order they are drawn.
    """
    through = math.prod(
        OPAQUE - puff_alpha(base, puff_density(puff_position(point, centre))) for centre in centres
    )
    return Alpha(OPAQUE - through)


@deal.pure
@deal.pre(lambda bits: 0 <= bits <= _DEPTH_FRACTION_BITS)
@deal.post(lambda result: result >= 0)
@deal.ensure(lambda bits, result: result.bit_length() == bits)
@deal.ensure(lambda bits, result: result & (result + 1) == 0)
def cleared_depth_bits(bits: BitCount) -> int:
    """The lowest `bits` bits, all set: the bits of a depth that puffs of one plane may differ in."""
    return (1 << bits) - 1


# The declarations that the vertex stage adds. PACK_ marks every name as the pack's, so that a later
# game version cannot bring a name of its own that collides.
_VERTEX_DECLARATIONS = GlslCode(f"""\
layout(location = 2) out vec2 packPuffPosition;

// The game's numbers for the two level faces of a cell, and how many vertices a face has.
const int PACK_FACE_DOWN = 0;
const int PACK_FACE_UP = 1;
const int PACK_FACE_VERTICES = 4;
const int PACK_LAST_FACE_VERTEX = PACK_FACE_VERTICES - 1;

const float PACK_PUFF_HALF_SIZE = {PUFF_HALF_SIZE!r};
const float PACK_CELL_CENTRE = {CELL_CENTRE!r};

// Outside the part of space the game draws, so a face whose vertices are all put here has no fragment.
const vec4 PACK_NOWHERE = vec4(2.0, 2.0, 2.0, 1.0);""")

# After the game has placed the vertex on its cell: the level faces are grown into puffs. A face seen
# from behind is wound the other way round, because the fancy pipeline culls faces that look away, and
# between the top and the bottom of the layer every level face does.
_VERTEX_PUFF = GlslCode("""\
    bool packIsPuff = (direction == PACK_FACE_DOWN || direction == PACK_FACE_UP) && !isInsideFace;
    bool packSeenFromBehind = (direction == PACK_FACE_UP) == (pos.y > 0.0);
    int packQuadVertex = packSeenFromBehind ? PACK_LAST_FACE_VERTEX - quadVertex : quadVertex;
    vec3 packFaceVertex = vertex((direction * PACK_FACE_VERTICES) + packQuadVertex);
    packPuffPosition = sign(packFaceVertex.xz - PACK_CELL_CENTRE);
    vec2 packCell = vec2(cellX, cellZ) + PACK_CELL_CENTRE + packPuffPosition * PACK_PUFF_HALF_SIZE;
    pos.xz = packCell * CellSize.xz + CloudOffset.xz;""")

# After the game has coloured the vertex. A level face seen from behind shows the other side of the
# cloud, so it takes that side's colour: the underside stays shaded and the top bright from wherever
# they are seen. Fast clouds are one face in the top colour and stay so.
_VERTEX_COLOUR = GlslCode("""\
    if (packIsPuff && packSeenFromBehind && !useTopColor) {
        vertexColor = faceColors[PACK_FACE_UP - direction] * CloudColor;
    }
    if (!packIsPuff) {
        gl_Position = PACK_NOWHERE;
    }""")

_FRAGMENT_DECLARATIONS = GlslCode(f"""\
layout(location = 2) in vec2 packPuffPosition;

const float PACK_CENTRE = {CENTRE!r};
const float PACK_EDGE = {EDGE!r};
const float PACK_FULL_CLOUD = {FULL_CLOUD!r};
const float PACK_OPAQUE = {OPAQUE!r};
const float PACK_DENSEST_ALPHA = {DENSEST_ALPHA!r};
const float PACK_SMOOTH_OFFSET = {_SMOOTH_OFFSET!r};
const float PACK_SMOOTH_SLOPE = {_SMOOTH_SLOPE!r};
const int PACK_CLEARED_DEPTH_BITS = {cleared_depth_bits(SHARED_DEPTH_BITS):#x};

float packFalloff(float reach) {{
    float t = min(abs(reach), PACK_EDGE);
    return PACK_FULL_CLOUD - t * t * (PACK_SMOOTH_OFFSET - PACK_SMOOTH_SLOPE * t);
}}

float packPuffDensity(vec2 position) {{
    return packFalloff(position.x) * packFalloff(position.y);
}}

float packPuffAlpha(float base, float density) {{
    return PACK_OPAQUE - pow(PACK_OPAQUE - min(base, PACK_DENSEST_ALPHA), density);
}}""")

# Before the game hands the colour to its phases, and after it has faded the alpha with distance, so
# that the faded alpha is what is shared out between the puffs and a distant cloud stays as even as a
# near one. Every phase gets the same alpha and the same depth.
_FRAGMENT_PUFF = GlslCode("""\
    color.a = packPuffAlpha(color.a, packPuffDensity(packPuffPosition));
    gl_FragDepth = intBitsToFloat(floatBitsToInt(gl_FragCoord.z) & ~PACK_CLEARED_DEPTH_BITS);""")

_ATTACHMENTS = (
    Attachment(VERTEX_SHADER, VERTEX_COLOUR_OUTPUT, Position.AFTER, _VERTEX_DECLARATIONS),
    Attachment(VERTEX_SHADER, VERTEX_POSITION, Position.AFTER, _VERTEX_PUFF),
    Attachment(VERTEX_SHADER, VERTEX_COLOUR, Position.AFTER, _VERTEX_COLOUR),
    Attachment(FRAGMENT_SHADER, FRAGMENT_COLOUR_INPUT, Position.AFTER, _FRAGMENT_DECLARATIONS),
    Attachment(FRAGMENT_SHADER, FRAGMENT_PHASES, Position.BEFORE, _FRAGMENT_PUFF),
)


@deal.pure
@deal.post(lambda result: result.name == NAME)
@deal.post(lambda result: {attachment.shader for attachment in result.attachments} == {VERTEX_SHADER, FRAGMENT_SHADER})
@deal.post(lambda result: len({(a.shader, a.place) for a in result.attachments}) == len(result.attachments))
@deal.post(lambda result: result.includes == ())
def effect() -> Effect:
    """The clouds effect: the code that turns the level faces of each cloud cell into a soft puff."""
    return Effect(NAME, _ATTACHMENTS)
