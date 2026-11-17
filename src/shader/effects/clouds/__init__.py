"""Clouds in the forms of real clouds: cirrus, altocumulus, cumulus and stratus, drawn from noise.

The game keeps its clouds as a grid of cells, 12 blocks wide and 4 high, each full or empty, and draws a
flat face for each full cell. No real cloud fits that grid, so the cells are not used as clouds here. The
cloud map of the pack fills every cell (shader.cloudmap), which makes the game send a face for every cell
of the layer: the layer becomes a sheet to draw on, at the height the game gives its clouds, hidden behind
terrain and faded with distance as the game does it. What is drawn on the sheet is worked out for each
pixel from where the pixel lies in the world.

Where a pixel lies. The cloud pipelines of 26.3 are given the Globals block (see the design), which holds
the block the camera is in, how far the camera is from that block's corner, and the game's clock. The
vertex stage hands on where a point of a face lies from the camera. Block, corner and point together are a
place in the world, kept as whole blocks and a rest so that it stays exact millions of blocks from the
origin, where one float would not.

Noise. A lattice lies over the world, and a hash of which corner it is gives each corner bits of its
own. A lattice is always a power of two of blocks wide, so that the corner a place belongs to comes from
shifting bits, which GLSL defines for a negative number where it does not define division. Two noises
are made of a lattice. Smooth noise gives each corner a level and blends the levels between corners.
Billow noise puts one round puff in each cell, its centre placed by the bits, and is how near the nearest
centre is. Several lattices, each half the size of the one before, are added up, the finer ones weaker:
the coarse ones give a cloud its outline and the fine ones its detail.

Forms. A form is a set of numbers for that sum (Shape). Cirrus is smooth noise on a lattice sixteen times
longer than wide, thin and soft. Altocumulus is small billows, which are its cloudlets. Cumulus is larger
billows gathered into heaps by its field. Stratus is smooth, very large and mostly cloud.

Which form where. One more smooth noise, on lattices of its own that do not drift, gives every place a
kind from 0 to 1. The forms sit along that range in the order thin to heavy, and a place has the form its
kind is nearest to. Between two forms the one fades into the other over a short stretch. The form of a
place never changes: the clock says only how far through its 24000 ticks it is, not which day it is, so
there is nothing to change the form by.

Drift. The clouds move west by DRIFT_SPAN blocks while the clock runs from 0 to 1. When the clock starts
again the clouds jump back by exactly that span, so the drifting noise repeats every DRIFT_SPAN blocks
from east to west and the jump lands every cloud on its copy. From north to south nothing repeats.

Height. The layer is a slab SLAB_HEIGHT blocks thick. A cloud stands in it from a flat base at the bottom
up to a top, a dome with steep sides that is highest where the cloud's coarse lattices are: cumulus up to
the whole slab, the other forms a fraction of it. The vertex stage lifts the game's top face to the top of
the slab, so a line of sight enters the slab at a face from either side, and the fragment stage follows the
line through the slab until it is under a top (seen_cloud). From below that shows the bases and, through
the gaps, the sides of the clouds behind; from above, the tops. The game still tests the face, not the
cloud, against the terrain, so a mountain inside the slab cuts the clouds off at the face.

Far away. A lattice narrower than a few pixels would flicker as the camera moves. Each lattice is faded
out as it nears NARROWEST_PIXELS pixels and is left at its middle level below that, which keeps the amount
of cloud the same and drops only the detail. From FLAT_FOOTPRINT on the clouds are drawn flat, since their
height cannot be made out.

Light. Light comes from above. The base of a thick cloud takes the game's colour of an underside, and the
colour rises to the game's colour of a top with the height at which the cloud is seen. A thin part of a
cloud lets the light through, so it takes the top's colour at any height.

What the game sends (CloudRenderer and RenderPipelines of 26.3) once every cell is full, and what is made
of it:

- Fancy clouds send the top face of a cell unless the camera is below the layer and the bottom face
  unless it is above. No cell has an empty neighbour, so no side face is sent. The nine cells around the
  camera are sent again with all six faces, marked as inside faces. Fast clouds send one bottom face per
  cell, marked to take the top colour.
- From below a pixel sees the bottom faces, from above the top faces. A face that looks away from the
  camera, which the game would cull, is turned to face it, and the cloud is drawn flat on it: the slab
  then lies between the camera and the face. With the camera inside the slab and above the game's own
  layer of 4 blocks, the game sends no bottom face, and nothing is drawn below the camera. Inside faces
  would draw the same sheet again and are not drawn.
- All but one of the ways the game draws clouds write depth wherever a face is, cloud or not. A pixel with
  next to no cloud is discarded, so that the clear parts of the sheet do not hide what is drawn after it.

Cost: a pixel whose line of sight enters under a thick base takes one noise for the kind and one column,
which is a noise per lattice of each form the place has and one for its field. A smooth noise is four
hashes of a few integer operations and a billow noise nine. A pixel that enters in a gap, near enough for
height to show, also takes a height at up to MARCH_STEPS + HALVINGS looks, each BODY_OCTAVES noises and a
field, and two more columns where it meets a cloud. No texture is read.

The maths lives twice: as the functions below, which the tests check, and as the GLSL that `effect`
attaches, which the game runs. GLSL cannot be run where the tests run, and Python cannot be run in the
game. Both are written from the same named constants, and each GLSL function mirrors the Python one of
the same name line for line, so that the two can be compared by eye. The hashes are 32 bit integer
arithmetic and agree to the bit. The rest is float arithmetic, 32 bit in GLSL and 64 bit here, and differs
by rounding.

Seen in the game: that the forms are drawn. Not yet seen: their height, and what it costs in frames. The
forms and the height were chosen from pictures drawn from the functions below, not from the game.
"""

import deal

from shader.effects import Attachment, Effect, EffectName, GlslCode, Place, Position
from shader.effects.clouds import forms, noise, sight
from shader.effects.clouds.forms import NO_GAIN
from shader.effects.clouds.sight import SLAB_HEIGHT
from shader.vanilla import ShaderPath

# What a build calls this effect.
NAME = EffectName("clouds")

VERTEX_SHADER = ShaderPath("core/clouds.vsh")

FRAGMENT_SHADER = ShaderPath("core/clouds.fsh")

# The lines of the game's clouds.vsh and clouds.fsh that the code attaches to.
VERTEX_COLOUR_OUTPUT = Place("layout(location = 1) out vec4 vertexColor;")

VERTEX_POSITION = Place("vec3 pos = (faceVertex * CellSize) + (vec3(cellX, 0, cellZ) * CellSize) + CloudOffset;")

VERTEX_COLOUR = Place("vertexColor = (useTopColor ? faceColors[1] : faceColors[direction]) * CloudColor;")

FRAGMENT_LAST_INCLUDE = Place("#include <minecraft:oit.glsl>")

FRAGMENT_COLOUR_INPUT = Place("layout(location = 1) in vec4 vertexColor;")

FRAGMENT_PHASES = Place("#ifdef OIT_ALPHA_ONLY")

# The game's own include that declares the Globals block.
GLOBALS_INCLUDE = GlslCode("#include <minecraft:globals.glsl>")

# The declarations that the vertex stage adds. PACK_ marks every name as the pack's, so that a later
# game version cannot bring a name of its own that collides.
_VERTEX_DECLARATIONS = GlslCode(f"""\
layout(location = 2) out vec3 packFromCamera;
layout(location = 3) out float packShade;
layout(location = 4) out float packRising;
layout(location = 5) out float packFlat;

// The game's numbers for the two level faces of a cell, and how many vertices a face has.
const int PACK_FACE_DOWN = 0;
const int PACK_FACE_UP = 1;
const int PACK_FACE_VERTICES = 4;
const int PACK_LAST_FACE_VERTEX = PACK_FACE_VERTICES - 1;

const float PACK_NO_GAIN = {NO_GAIN!r};
const float PACK_SLAB_HEIGHT = {SLAB_HEIGHT!r};
const float PACK_YES = 1.0;
const float PACK_NO = 0.0;

// Outside the part of space the game draws, so a face whose vertices are all put here has no fragment.
const vec4 PACK_NOWHERE = vec4(2.0, 2.0, 2.0, 1.0);""")

# After the game has placed the vertex on its cell. The top face is lifted to the top of the slab, so that
# a line of sight from above enters the slab where its face is, as one from below does at the bottom face.
# A face seen from behind is wound the other way round, because the fancy pipeline culls faces that look
# away, and inside the slab one of the two always does. Seen from behind, the slab lies between the camera
# and the face and not beyond it, so there the cloud is drawn flat.
_VERTEX_FACE = GlslCode("""\
    bool packIsLevel = (direction == PACK_FACE_DOWN || direction == PACK_FACE_UP) && !isInsideFace;
    if (direction == PACK_FACE_UP) {
        pos.y += PACK_SLAB_HEIGHT - CellSize.y;
    }
    bool packSeenFromBehind = (direction == PACK_FACE_UP) == (pos.y > 0.0);
    int packQuadVertex = packSeenFromBehind ? PACK_LAST_FACE_VERTEX - quadVertex : quadVertex;
    vec3 packFaceVertex = vertex((direction * PACK_FACE_VERTICES) + packQuadVertex);
    pos.xz = (packFaceVertex.xz + vec2(cellX, cellZ)) * CellSize.xz + CloudOffset.xz;
    packFromCamera = pos;
    packRising = direction == PACK_FACE_DOWN ? PACK_YES : PACK_NO;
    packFlat = packSeenFromBehind ? PACK_YES : PACK_NO;""")

# After the game has coloured the vertex. Every level face takes the game's colour of a top, and the
# fragment stage darkens it by how little light the seen part of the cloud gets, down to the shade that
# takes the top's colour to the game's colour of an underside. Fast clouds are one face in the top colour,
# which the game means to be unshaded, and stay so.
_VERTEX_COLOUR = GlslCode("""\
    packShade = PACK_NO_GAIN;
    if (packIsLevel && !useTopColor) {
        vertexColor = faceColors[PACK_FACE_UP] * CloudColor;
        packShade = faceColors[PACK_FACE_DOWN].r / faceColors[PACK_FACE_UP].r;
    }
    if (!packIsLevel) {
        gl_Position = PACK_NOWHERE;
    }""")

# What the vertex stage hands on, then the noise, the forms and the line of sight, each built on the one before.
_FRAGMENT_DECLARATIONS = GlslCode(
    """\
layout(location = 2) in vec3 packFromCamera;
layout(location = 3) in float packShade;
layout(location = 4) in float packRising;
layout(location = 5) in float packFlat;

"""
    + noise.GLSL
    + "\n"
    + forms.GLSL
    + "\n"
    + sight.GLSL
)

# Before the game hands the colour to its phases, and after it has faded the alpha with distance, so
# that a distant cloud thins as the game's own does. Every phase gets the same alpha. The camera is at
# its block less CameraOffset (GlobalSettingsUniform of 26.3 stores the block's corner less the camera),
# so a fragment is that much further from the block's corner than from the camera. The footprint is
# taken along whichever screen axis covers more of the layer.
_FRAGMENT_CLOUD = GlslCode("""\
    vec2 packPixel = fwidth(packFromCamera.xz);
    float packFootprint = max(packPixel.x, packPixel.y);
    bool packDrawnFlat = packFlat > PACK_HALF || packFootprint >= PACK_FLAT_FOOTPRINT;
    vec2 packSeen = packSeenCloud(
        CameraBlockPos.xz, packFromCamera.xz - CameraOffset.xz,
        packSightTravel(packFromCamera.xz, abs(packFromCamera.y)), packRising > PACK_HALF,
        packDrawnFlat ? PACK_NO_STEPS : PACK_MARCH_STEPS, GameTime, packFootprint);
    color.rgb *= packLightGain(packShade, packLift(packSeen.x, packSeen.y));
    color.a = packCloudAlpha(color.a, packSeen.y);
    if (color.a < PACK_FAINTEST_ALPHA) {
        discard;
    }""")

_ATTACHMENTS = (
    Attachment(VERTEX_SHADER, VERTEX_COLOUR_OUTPUT, Position.AFTER, _VERTEX_DECLARATIONS),
    Attachment(VERTEX_SHADER, VERTEX_POSITION, Position.AFTER, _VERTEX_FACE),
    Attachment(VERTEX_SHADER, VERTEX_COLOUR, Position.AFTER, _VERTEX_COLOUR),
    Attachment(FRAGMENT_SHADER, FRAGMENT_LAST_INCLUDE, Position.AFTER, GLOBALS_INCLUDE),
    Attachment(FRAGMENT_SHADER, FRAGMENT_COLOUR_INPUT, Position.AFTER, _FRAGMENT_DECLARATIONS),
    Attachment(FRAGMENT_SHADER, FRAGMENT_PHASES, Position.BEFORE, _FRAGMENT_CLOUD),
)


@deal.pure
@deal.post(lambda result: result.name == NAME)
@deal.post(lambda result: {attachment.shader for attachment in result.attachments} == {VERTEX_SHADER, FRAGMENT_SHADER})
@deal.post(lambda result: len({(a.shader, a.place) for a in result.attachments}) == len(result.attachments))
@deal.post(lambda result: result.includes == ())
def effect() -> Effect:
    """The clouds effect: the code that draws the forms of cloud on the level faces of the cloud layer."""
    return Effect(NAME, _ATTACHMENTS)
