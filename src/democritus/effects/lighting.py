"""The colour of light: sun against shade, fire against moonlight, and bright light that rolls off.

The game draws a 16 by 16 light map once a frame with core/lightmap.fsh, and every surface in the world
reads its light from that texture by its block light level and its sky light level. A change here costs
nothing per pixel of the screen. It is also the only place shade can live: the terrain shader is told no
surface normal and no sun direction, so the sky light level is all a surface knows about being in shade.

The game's own steps stay where they are: ambient light, night vision, the boss darkening, the Darkness
effect and the brightness slider. Three things change around them.

Sky light is split into sun and shade. Only a surface under the open sky by day gets the sun, which is
warm and a little stronger than the game's. The sun's share falls off fast with the sky light level and
with the game's sky factor, so that under a tree, in a doorway, in rain and at night the light is the
cool blue of the sky alone. The game's own sky light colour is kept as a factor, so its night blue and
its rain still come through.

Block light is the game's own colour seen through a fire tint that takes away green and blue: much at
the dim edge of a torch's reach, little beside the flame. The tint only takes away, so the light is never
bluer than the game's, whatever tint the game gives a dimension.

The sum is rolled off toward white above a knee, where the game cuts it off at white. Below the knee
nothing changes, so the dark of a cave is as the game has it.

The maths lives twice: as the functions below, which the tests check, and as the GLSL that `effect`
attaches, which the game runs. GLSL cannot be run where the tests run, and Python cannot be run in the
game. Both are written from the same named constants, and each GLSL function mirrors the Python one of
the same name line for line, so that the two can be compared by eye.

Not yet seen in the game: every colour below was chosen from a rendering of the light map, not from the
world.
"""

import math
from dataclasses import dataclass
from typing import NewType

import deal

from democritus.effects import Attachment, Effect, EffectName, ExactLine, GlslCode, Landmark, Position
from democritus.vanilla import ShaderPath

# A block or sky light level as lightmap.fsh holds it: the level from 0 to 15, divided by 15.
LightLevel = NewType("LightLevel", float)

# How strong one colour channel of light is. 1 is the most a texel of the light map can hold.
Intensity = NewType("Intensity", float)

# The game's SkyFactor: how strong the sky light is, from the time of day and the weather.
SkyFactor = NewType("SkyFactor", float)

# How much of something there is, from 0 for none to 1 for all.
Share = NewType("Share", float)

# The power a light level is raised to.
Exponent = NewType("Exponent", float)

# The name of a constant in the GLSL code.
GlslName = NewType("GlslName", str)


@dataclass(frozen=True)
class Colour:
    """A colour of light, as the strength of its red, green and blue."""

    red: Intensity
    green: Intensity
    blue: Intensity

    def mixed(self, other: "Colour", share: Share) -> "Colour":
        """This colour moved toward another by a share, as GLSL's mix does."""
        return Colour(
            Intensity(self.red + (other.red - self.red) * share),
            Intensity(self.green + (other.green - self.green) * share),
            Intensity(self.blue + (other.blue - self.blue) * share),
        )

    def warmer_than(self, other: "Colour") -> bool:
        """Whether this colour has more red for its blue than another.

        The products are compared, not the ratios, so that a colour with no blue still has an answer.
        A colour is bluer than another exactly when the other is warmer than it.
        """
        return self.red * other.blue > other.red * self.blue

    def within(self, weakest: "Colour", strongest: "Colour") -> bool:
        """Whether each channel of this colour lies from that channel of one colour to that of another."""
        return (
            weakest.red <= self.red <= strongest.red
            and weakest.green <= self.green <= strongest.green
            and weakest.blue <= self.blue <= strongest.blue
        )


NAME = EffectName("lighting")
LIGHTMAP = ShaderPath("core/lightmap.fsh")

# The functions go before main. The rest replace or follow the game's own steps, so they attach to those exact
# lines: if Mojang changes one of them, the tint may no longer fit, and the build stops instead of guessing.
MAIN = Landmark.MAIN
SKY_LIGHT = ExactLine("color += lightmapInfo.SkyLightColor * sky_brightness;")
BLOCK_LIGHT = ExactLine(
    "vec3 BlockLightColor = mix(lightmapInfo.BlockLightTint, vec3(1.0), 0.9 * parabolicMixFactor(block_level));"
)
CLAMP = ExactLine("color = clamp(color, 0.0, 1.0);")

# The sky factor of a clear night and of a clear day, as the day timeline of 26.3 sets them. Rain brings
# a day down to 0.76 and thunder to 0.60, so bad weather loses part of the sun and keeps the rest.
NIGHT_SKY_FACTOR = SkyFactor(0.24)
DAY_SKY_FACTOR = SkyFactor(1.0)

# Sky light loses one level for each block it travels sideways, and for each layer of leaves. At 8 a
# surface one block in from the open keeps 58 % of the sun, three blocks in 17 %, and five blocks in
# almost none: the edge of a shadow is soft, and two steps into a doorway it is shade.
SUN_REACH = Exponent(8.0)

# Sunlight is warm, and stronger than 1: the shoulder brings it back to just under white. Being over the
# knee is what lets a torch in daylight still add something.
SUN = Colour(Intensity(1.30), Intensity(1.22), Intensity(1.08))

# Light from the blue sky alone, without the sun: what lights shade by day and everything by night.
# It is weaker than the sun in every channel, so shade is dimmer as well as cooler.
SHADE = Colour(Intensity(0.70), Intensity(0.85), Intensity(1.04))

# What is left of block light at the dim edge of its reach, and beside its source. Red is kept whole, so
# the tint can only make light warmer; a flame is a soft warm white and its far glow a deep amber.
EMBER = Colour(Intensity(1.0), Intensity(0.62), Intensity(0.30))
FLAME = Colour(Intensity(1.0), Intensity(0.92), Intensity(0.78))

# Light up to the knee is left as it is, which keeps caves and night as dark as the game has them and
# no darker.
KNEE = Intensity(0.6)

# The curve above the knee is the parabola that leaves the knee at slope 1 and is level at 1. Those two
# ask for this white point and this strength, so the knee is the one number to choose.
WHITE = Intensity(1.0)
WHITE_POINT = Intensity(2 * WHITE - KNEE)
ROLL_OFF = Intensity(1 / (2 * (WHITE_POINT - KNEE)))

_NO_LIGHT = Intensity(0.0)
_NONE = Share(0.0)
_ALL = Share(1.0)

# The two numbers of the smoothstep curve t * t * (3 - 2 * t), which GLSL has built in.
_SMOOTH_OFFSET = 3
_SMOOTH_SLOPE = 2


@deal.pure
# The game gives numbers. Without this a NaN would pass every comparison below and come out as a NaN.
@deal.pre(lambda low, high, value: math.isfinite(low) and math.isfinite(high) and math.isfinite(value))
@deal.pre(lambda low, high, value: low < high)
@deal.post(lambda result: _NONE <= result <= _ALL)
def _smoothstep(low: SkyFactor, high: SkyFactor, value: SkyFactor) -> Share:
    """How far a value is from low to high, eased at both ends, as GLSL's smoothstep gives it."""
    along = min(max((value - low) / (high - low), _NONE), _ALL)
    return Share(along * along * (_SMOOTH_OFFSET - _SMOOTH_SLOPE * along))


@deal.pure
@deal.pre(lambda sky_level, sky_factor: _NONE <= sky_level <= _ALL)
@deal.pre(lambda sky_level, sky_factor: math.isfinite(sky_factor))
@deal.post(lambda result: _NONE <= result <= _ALL)
@deal.ensure(lambda sky_level, sky_factor, result: sky_level == _ALL or result < _ALL)
def sun_share(sky_level: LightLevel, sky_factor: SkyFactor) -> Share:
    """How much of its sky light a surface gets from the sun, the rest being the light of the sky alone."""
    open_sky = sky_level**SUN_REACH
    daylight = _smoothstep(NIGHT_SKY_FACTOR, DAY_SKY_FACTOR, sky_factor)
    return Share(open_sky * daylight)


@deal.pure
@deal.pre(lambda sky_level, sky_factor: _NONE <= sky_level <= _ALL)
@deal.pre(lambda sky_level, sky_factor: math.isfinite(sky_factor))
# The sun is stronger than the shade in every channel and most of all in red, so a tint between the two
# is no warmer than the sun. Held channel by channel because that stays exact in floating point.
@deal.post(lambda result: result.within(SHADE, SUN))
def sky_tint(sky_level: LightLevel, sky_factor: SkyFactor) -> Colour:
    """The colour that the game's sky light is seen through at a sky light level."""
    return SHADE.mixed(SUN, sun_share(sky_level, sky_factor))


@deal.pure
@deal.pre(lambda block_level: _NONE <= block_level <= _ALL)
@deal.post(lambda result: result.red == WHITE and result.blue <= result.green <= result.red)
def fire_tint(block_level: LightLevel) -> Colour:
    """The colour that the game's block light is seen through at a block light level."""
    return EMBER.mixed(FLAME, Share(block_level))


@deal.pure
@deal.pre(lambda light: not math.isnan(light))
@deal.post(lambda result: _NO_LIGHT <= result <= WHITE)
@deal.ensure(lambda light, result: result <= max(light, _NO_LIGHT))
def shoulder(light: Intensity) -> Intensity:
    """One channel of the summed light, rolled off toward white where the game would cut it off."""
    lit = min(max(light, _NO_LIGHT), WHITE_POINT)
    over = max(lit - KNEE, _NO_LIGHT)
    return Intensity(lit - over * over * ROLL_OFF)


@deal.pure
def _float(number: Intensity | SkyFactor | Exponent) -> GlslCode:
    # repr gives the shortest text that reads back as the same number, and always with a point or an
    # exponent, which is what makes it a float to GLSL.
    return GlslCode(repr(float(number)))


@deal.pure
def _vec3(colour: Colour) -> GlslCode:
    return GlslCode(f"vec3({_float(colour.red)}, {_float(colour.green)}, {_float(colour.blue)})")


@deal.pure
def _constant(glsl_type: GlslCode, name: GlslName, value: GlslCode) -> GlslCode:
    return GlslCode(f"const {glsl_type} {name} = {value};\n")


_FLOAT = GlslCode("float")
_VEC3 = GlslCode("vec3")

# Every name starts with pack, so that none can meet a name of the game's or of a later version's.
_CONSTANTS = GlslCode(
    _constant(_FLOAT, GlslName("PACK_NIGHT_SKY_FACTOR"), _float(NIGHT_SKY_FACTOR))
    + _constant(_FLOAT, GlslName("PACK_DAY_SKY_FACTOR"), _float(DAY_SKY_FACTOR))
    + _constant(_FLOAT, GlslName("PACK_SUN_REACH"), _float(SUN_REACH))
    + _constant(_VEC3, GlslName("PACK_SUN"), _vec3(SUN))
    + _constant(_VEC3, GlslName("PACK_SHADE"), _vec3(SHADE))
    + _constant(_VEC3, GlslName("PACK_EMBER"), _vec3(EMBER))
    + _constant(_VEC3, GlslName("PACK_FLAME"), _vec3(FLAME))
    + _constant(_FLOAT, GlslName("PACK_KNEE"), _float(KNEE))
    + _constant(_FLOAT, GlslName("PACK_WHITE_POINT"), _float(WHITE_POINT))
    + _constant(_FLOAT, GlslName("PACK_ROLL_OFF"), _float(ROLL_OFF))
)

# Each function is the Python function of the same name, line for line: sun_share, sky_tint, fire_tint
# and shoulder. A change to one is a change to the other.
_FUNCTIONS = GlslCode(
    """
float packSunShare(float skyLevel, float skyFactor) {
    float openSky = pow(skyLevel, PACK_SUN_REACH);
    float daylight = smoothstep(PACK_NIGHT_SKY_FACTOR, PACK_DAY_SKY_FACTOR, skyFactor);
    return openSky * daylight;
}

vec3 packSkyTint(float skyLevel, float skyFactor) {
    return mix(PACK_SHADE, PACK_SUN, packSunShare(skyLevel, skyFactor));
}

vec3 packFireTint(float blockLevel) {
    return mix(PACK_EMBER, PACK_FLAME, blockLevel);
}

vec3 packShoulder(vec3 light) {
    vec3 lit = min(max(light, 0.0), PACK_WHITE_POINT);
    vec3 over = max(lit - PACK_KNEE, 0.0);
    return lit - over * over * PACK_ROLL_OFF;
}

"""
)

_TINTED_SKY_LIGHT = GlslCode(
    "    color += lightmapInfo.SkyLightColor * sky_brightness * packSkyTint(sky_level, lightmapInfo.SkyFactor);\n"
)
_TINTED_BLOCK_LIGHT = GlslCode("    BlockLightColor *= packFireTint(block_level);\n")
_ROLLED_OFF = GlslCode("    color = packShoulder(color);\n")


@deal.pure
@deal.post(lambda result: result.name == NAME)
@deal.post(lambda result: all(attachment.shader == LIGHTMAP for attachment in result.attachments))
@deal.post(lambda result: not result.includes)
def effect() -> Effect:
    """The lighting, as code for the game's lightmap.fsh.

    The block light keeps the game's line and is tinted after it, and the game's clamp stays behind the
    shoulder: what the game does is still there to read, and a later version that changes either line is
    noticed by the composer.
    """
    return Effect(
        NAME,
        (
            Attachment(LIGHTMAP, MAIN, Position.BEFORE, GlslCode(_CONSTANTS + _FUNCTIONS)),
            Attachment(LIGHTMAP, SKY_LIGHT, Position.INSTEAD, _TINTED_SKY_LIGHT),
            Attachment(LIGHTMAP, BLOCK_LIGHT, Position.AFTER, _TINTED_BLOCK_LIGHT),
            Attachment(LIGHTMAP, CLAMP, Position.BEFORE, _ROLLED_OFF),
        ),
    )
