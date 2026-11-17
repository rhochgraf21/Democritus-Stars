"""The sky as a gradient: the fog colour at the horizon, rising smoothly to the sky colour overhead.

The game draws the sky as a flat disc above the player in one colour, ColorModulator, and blends it toward
FogColor by the distance along the disc. That gives a narrow band at the horizon under a flat sky. Here the
blend goes by the height of the direction a pixel looks in instead, the way air does it: more air lies along
a low line of sight, so the sky is palest at the horizon and deepest straight up.

What was read from 26.3's SkyRenderer, and what the code leans on:

- The sky disc lies 16 above the player and reaches 512 out, as an octagon. Its rim is therefore never lower
  than a height of 16 / 512 and never higher than 16 / 473 (the middle of an edge). Below the rim the game
  shows the colour it cleared the screen with, which is the fog colour.
- The same program draws the void, a black disc 16 below, when the player is under the horizon height of the
  world. Its ColorModulator is black. It has to stay dark underneath and still meet the fog at its rim, as
  the game's own distance fog made it.
- sky.vsh and sky.fsh are given Fog, DynamicTransforms and Projection, and no time: nothing here can move.
- At sunrise and sunset the game changes FogColor and ColorModulator itself. Both are used as given.

The gradient is one GLSL function in an include file of its own, so that the water can reflect exactly the
sky that is drawn: see gradient_include, GRADIENT_INCLUDE_LINE and GRADIENT_FUNCTION.

The maths lives twice: as GLSL, which the game runs, and as Python (fog_share and sky_colour), which the
tests run, because GLSL cannot be run where the tests are. Both are written from the same named constants
and the GLSL function mirrors the Python line for line. Change one, change the other.

Not yet seen in the game: the look itself, that no seam shows at the rim of the disc, and the void.
"""

from dataclasses import dataclass
from typing import NewType

import deal

from democritus.effects import Attachment, Effect, EffectName, ExactLine, GlslCode, Landmark, Position
from democritus.vanilla import ShaderFile, ShaderPath, ShaderText

# One channel of a colour as the shaders hold it, from 0 for none to 1 for full.
Intensity = NewType("Intensity", float)

# How far up a direction of view points: the y of the direction at length 1, so 0 is level and 1 is straight up.
Height = NewType("Height", float)

# How much of a colour of the sky is the fog colour, from 0 for none to 1 for all of it.
FogShare = NewType("FogShare", float)


@dataclass(frozen=True)
class Colour:
    """A colour as the shaders hold it: red, green and blue."""

    red: Intensity
    green: Intensity
    blue: Intensity


NO_LIGHT = Intensity(0.0)
FULL_LIGHT = Intensity(1.0)

LEVEL = Height(0.0)
STRAIGHT_UP = Height(1.0)
STRAIGHT_DOWN = Height(-1.0)

ALL_FOG = FogShare(1.0)
NO_FOG = FogShare(0.0)

# The height at and below which the sky is the fog colour. Not 0: the rim of the game's sky disc sits between
# 16 / 512 and 16 / 473 (0.031 to 0.034), and below it the screen already is the fog colour. A gradient that
# only reached the fog colour at 0 would still be a shade off at the rim, and show the rim as a line.
HORIZON_HEIGHT = Height(0.04)

# How high the pale of the horizon reaches: about half of the fog colour is left this far up the way from the
# horizon to straight up. Chosen by eye from rendered strips of noon, sunset and night: at 0.2 three quarters
# of the change is over 20 degrees up, as in a photograph of a clear day. 0.16 left a band, 0.25 a haze.
HAZE_REACH = Height(0.2)

# How far a result may stray from the exact one through the rounding of floating point numbers.
ROUNDING = Intensity(1e-12)

# The same numbers, as GLSL writes them. repr gives the shortest text that reads back as the same number.
_GLSL_HORIZON_HEIGHT = repr(float(HORIZON_HEIGHT))
_GLSL_HAZE_REACH = repr(float(HAZE_REACH))

NAME = EffectName("sky")

_VERTEX_SHADER = ShaderPath("core/sky.vsh")
_FRAGMENT_SHADER = ShaderPath("core/sky.fsh")

# Where the gradient's include file sits in the pack.
GRADIENT_INCLUDE_PATH = ShaderPath("include/pack_sky.glsl")

# The line that brings the gradient into a shader.
GRADIENT_INCLUDE_LINE = GlslCode("#include <minecraft:pack_sky.glsl>")

# The GLSL function that gives the colour of the sky: vec3 f(vec3 skyColour, vec3 fogColour, float height).
# height is the y of the direction of view at length 1. At the horizon and below, it gives the fog colour.
GRADIENT_FUNCTION = GlslCode("pack_sky_colour")

_GRADIENT_TEXT = ShaderText(
    f"""#ifndef PACK_SKY_GLSL
#define PACK_SKY_GLSL

// Written from democritus.effects.sky, which holds the same maths in Python for the tests. Change both.
const float PACK_SKY_HORIZON_HEIGHT = {_GLSL_HORIZON_HEIGHT};
const float PACK_SKY_HAZE_REACH = {_GLSL_HAZE_REACH};

vec3 {GRADIENT_FUNCTION}(vec3 skyColour, vec3 fogColour, float height) {{
    float rise = clamp((height - PACK_SKY_HORIZON_HEIGHT) / (1.0 - PACK_SKY_HORIZON_HEIGHT), 0.0, 1.0);
    float lift = rise * rise;
    float reach = PACK_SKY_HAZE_REACH * PACK_SKY_HAZE_REACH;
    float fogShare = reach * (1.0 - lift) / (reach + lift);
    return skyColour * (1.0 - fogShare) + fogColour * fogShare;
}}

#endif
"""
)

# The direction of view goes to the fragment stage in a varying of its own. The game's sky shaders use
# locations 0 and 1, and the two stages must agree on the number.
_DIRECTION_VARYING = "layout(location = 2) {way} vec3 packSkyDirection;"

# Only the colour is replaced, so only it is an exact line: the rest needs nothing of the game's text but its shape.
_FRAGMENT_COLOUR = ExactLine(
    "fragColor = apply_fog(ColorModulator, sphericalVertexDistance, cylindricalVertexDistance,"
    " 0.0, FogSkyEnd, FogSkyEnd, FogSkyEnd, FogColor);"
)

# Position is the vertex on the disc, in the world's axes with the player at the middle: the model-view
# matrix only turns it to the camera. So it is the direction of view before that turn, which is the one whose
# y is a height. It is passed on at full length and made length 1 per pixel, because a direction of length 1
# does not stay length 1 when it is blended across a triangle.
_VERTEX_DIRECTION = GlslCode("    packSkyDirection = Position;")

# abs: the void disc lies below the player, so its directions point down. Mirrored, it gets the game's black
# straight down and the fog colour at its rim, as the game's own fog, which went by distance, gave it. Without
# it the whole void would be the fog colour: a bright floor where the game wants darkness.
# FogColor.a: the game's apply_fog scales its fog by it, so the gradient is scaled the same way.
# The alpha stays ColorModulator's, as the game wrote it.
_FRAGMENT_GRADIENT = GlslCode(
    f"""    float packSkyHeight = abs(normalize(packSkyDirection).y);
    vec3 packSkyGradient = {GRADIENT_FUNCTION}(ColorModulator.rgb, FogColor.rgb, packSkyHeight);
    fragColor = vec4(mix(ColorModulator.rgb, packSkyGradient, FogColor.a), ColorModulator.a);"""
)

_ATTACHMENTS = (
    Attachment(_VERTEX_SHADER, Landmark.LAST_OUTPUT, Position.AFTER, GlslCode(_DIRECTION_VARYING.format(way="out"))),
    Attachment(_VERTEX_SHADER, Landmark.END_OF_MAIN, Position.BEFORE, _VERTEX_DIRECTION),
    Attachment(_FRAGMENT_SHADER, Landmark.LAST_INCLUDE, Position.AFTER, GRADIENT_INCLUDE_LINE),
    Attachment(_FRAGMENT_SHADER, Landmark.LAST_INPUT, Position.AFTER, GlslCode(_DIRECTION_VARYING.format(way="in"))),
    Attachment(_FRAGMENT_SHADER, _FRAGMENT_COLOUR, Position.INSTEAD, _FRAGMENT_GRADIENT),
)


@deal.has()
def _is_intensity(channel: Intensity) -> bool:
    """Whether a channel lies from none to full, which a number that is not a number does not."""
    return NO_LIGHT <= channel <= FULL_LIGHT


@deal.has()
def _is_colour(colour: Colour) -> bool:
    """Whether every channel of a colour lies from none to full."""
    return all(_is_intensity(channel) for channel in (colour.red, colour.green, colour.blue))


@deal.has()
def _is_between(channel: Intensity, one: Intensity, other: Intensity) -> bool:
    """Whether a channel lies between two others, give or take the rounding."""
    return min(one, other) - ROUNDING <= channel <= max(one, other) + ROUNDING


@deal.has()
@deal.pre(lambda height: STRAIGHT_DOWN <= height <= STRAIGHT_UP)
@deal.ensure(lambda height, result: NO_FOG <= result <= ALL_FOG)
@deal.ensure(lambda height, result: height > HORIZON_HEIGHT or result == ALL_FOG)
@deal.ensure(lambda height, result: height < STRAIGHT_UP or result == NO_FOG)
def fog_share(height: Height) -> FogShare:
    """How much of the sky's colour is the fog colour, for a direction of view of this height.

    All of it at the horizon and below, none of it straight up, and never more for a higher direction. The
    way up is squared so that the gradient leaves the horizon level, without an edge where it starts; the
    division gives the long tail of a real sky, which goes on deepening all the way up.
    """
    rise = min(max((height - HORIZON_HEIGHT) / (STRAIGHT_UP - HORIZON_HEIGHT), LEVEL), STRAIGHT_UP)
    lift = rise * rise
    reach = HAZE_REACH * HAZE_REACH
    return FogShare(reach * (ALL_FOG - lift) / (reach + lift))


@deal.has()
@deal.pre(lambda sky, fog, share: _is_intensity(sky) and _is_intensity(fog))
@deal.pre(lambda sky, fog, share: NO_FOG <= share <= ALL_FOG)
@deal.ensure(lambda sky, fog, share, result: _is_between(result, sky, fog))
@deal.ensure(lambda sky, fog, share, result: share != ALL_FOG or result == fog)
@deal.ensure(lambda sky, fog, share, result: share != NO_FOG or result == sky)
def _mixed(sky: Intensity, fog: Intensity, share: FogShare) -> Intensity:
    """One channel of the sky, moved toward the same channel of the fog by a share."""
    # Not sky + (fog - sky) * share: that form does not give back fog exactly at a share of 1.
    return Intensity(sky * (ALL_FOG - share) + fog * share)


@deal.has()
@deal.pre(lambda sky, fog, height: _is_colour(sky) and _is_colour(fog))
@deal.pre(lambda sky, fog, height: STRAIGHT_DOWN <= height <= STRAIGHT_UP)
@deal.ensure(lambda sky, fog, height, result: height > HORIZON_HEIGHT or result == fog)
@deal.ensure(lambda sky, fog, height, result: height < STRAIGHT_UP or result == sky)
@deal.ensure(
    lambda sky, fog, height, result: (
        _is_between(result.red, sky.red, fog.red)
        and _is_between(result.green, sky.green, fog.green)
        and _is_between(result.blue, sky.blue, fog.blue)
    )
)
def sky_colour(sky: Colour, fog: Colour, height: Height) -> Colour:
    """The colour of the sky in a direction of view of this height: what the GLSL gradient function gives."""
    share = fog_share(height)
    return Colour(
        _mixed(sky.red, fog.red, share),
        _mixed(sky.green, fog.green, share),
        _mixed(sky.blue, fog.blue, share),
    )


@deal.has()
@deal.ensure(lambda result: result.path == GRADIENT_INCLUDE_PATH and GRADIENT_FUNCTION in result.text)
def gradient_include() -> ShaderFile:
    """The include file that holds the gradient function, for an effect that draws or reflects the sky."""
    return ShaderFile(GRADIENT_INCLUDE_PATH, _GRADIENT_TEXT)


@deal.has()
@deal.ensure(lambda result: result.name == NAME and result.includes == (gradient_include(),))
@deal.ensure(lambda result: {a.shader for a in result.attachments} == {_VERTEX_SHADER, _FRAGMENT_SHADER})
def effect() -> Effect:
    """The sky effect: the gradient in place of the game's fog by distance along the sky disc."""
    return Effect(NAME, _ATTACHMENTS, (gradient_include(),))
