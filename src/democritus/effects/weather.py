"""How overcast the sky is, estimated from the fog colour and the sky light of the game.

The game tells no shader that it is raining, so the weather is read off two values that it does send.
What 26.3 does to them (WeatherAttributes, the day timeline, AtmosphericFogEnvironment, lightmap.fsh):

- The fog colour is the biome's (#c0d8ff in all but a few) times the hour (white by day, #0f0f16 at night)
  times the weather: rain multiplies it by (0.5, 0.5, 0.6), thunder by (0.25, 0.25, 0.3). So rain does not
  grey the fog. It darkens it, and leaves it a little bluer.
- The sky light is blended toward its night value, by 0.3125 in rain and 0.527 in thunder. Rain at noon
  lights the world exactly as a clear dusk does, so the sky light alone cannot tell them apart.
- The two fall at different hours. At dusk the sky light starts down 600 ticks before the fog colour does,
  and at dawn it comes back later, so in clear weather the fog is never darker than the sky light allows.
  Rain darkens the fog at once and the sky light only a little. That gap is the estimate.

Where the two values say nothing, the estimate is 0, "not overcast", so that nothing is thickened by mistake:
at night (the sky light sits at its floor while the fog colour still moves), and wherever the fog is not the
tint of the overworld's air (most water, lava, the Nether, the End, the pale garden).

The maths lives twice, as `overcast` here and as GLSL in the include file. GLSL cannot be run where the
tests run, so the Python one is what the tests hold to the game's numbers, and the GLSL one is written from
the same constants, line for line, and held to the Python one by a test that reads it back.
"""

from dataclasses import dataclass
from typing import NewType

import deal

from democritus.effects import Effect, EffectName, GlslCode
from democritus.vanilla import ShaderFile, ShaderPath, ShaderText

# A share of full, from 0 to 1: one channel of a colour as a shader holds it, or a strength.
Level = NewType("Level", float)

# The red of a colour over its blue: under 1 is bluer than grey, over 1 is warmer.
Warmth = NewType("Warmth", float)


@dataclass(frozen=True)
class Colour:
    """A colour as the game gives it to a shader: three levels, not yet made linear."""

    red: Level
    green: Level
    blue: Level


_NONE = Level(0.0)
_FULL = Level(1.0)

# The least a colour's blue counts as when it divides: one step of an 8-bit channel. A black fog (blindness)
# or a black light then has a warmth of 0 rather than no warmth at all.
_BLACK = Level(1 / 255)

# The air of the overworld: its fog's warmth runs from 0.59 (rain at dusk) to 0.76 (a clear noon), whatever
# the render distance. Water (0.1), the End (1.0) and the Nether's biomes (0.39, and 0.93 up) lie outside.
_AIR_WARMTH = Warmth(0.67)
_AIR_TOLERANCE = Warmth(0.19)
# The estimate fades out over this much past the tolerance rather than stopping, so that turning toward a
# sunset, which warms the fog, does not switch it off between two frames.
_AIR_EDGE = Warmth(0.05)

# The warmth of the light map at full sky light. Night is #7a7aff at a factor of 0.24 over the ambient
# #0a0a0a, which is 0.55 in 8 bits at any brightness setting; a little above, so that night reads as none.
_NIGHT_WARMTH = Warmth(0.56)
_DAY_WARMTH = Warmth(1.0)
# The overworld's sky light is never warmer than white. The Nether's light map is its ambient #302821,
# 1.45, and its biomes' fog colours blend through the air's tint at their borders.
_WARMEST_SKY = Warmth(1.2)

# The blue of the fog in clear weather when the sky light reaches its night floor. From there the clear
# fog's blue rises about one for one with the daylight until it is full: worked through the game's dusk and
# dawn, the clear fog is never more than 0.03 darker than this line.
_NIGHTFALL_FOG = Level(0.355)

# How much darker than clear the fog has to be. Under the first it is clear: a dappled forest's own fog is
# 0.11 darker. Rain is 0.40 darker at noon and 0.27 at dusk, when it also dims the sky light the fog is
# held against, so from the second on it is rain. Rain and thunder both read as 1, and the 100 ticks over
# which rain sets in read as the ramp between.
_CLEAR_DARKENING = Level(0.15)
_RAIN_DARKENING = Level(0.27)

# The daylight under which the estimate fades to nothing. The sky light says less and less about the hour
# as it nears its floor, and at the floor it says nothing.
_SURE_DAYLIGHT = Level(0.15)

NAME = EffectName("weather")

_INCLUDE_FILE = "pack_weather.glsl"

# Where the pack keeps the estimate.
INCLUDE_PATH = ShaderPath(f"include/{_INCLUDE_FILE}")

# The line that a shader needs, above its `main`, before it calls the estimate.
INCLUDE_LINE = GlslCode(f"#include <minecraft:{_INCLUDE_FILE}>")

# The name of the estimate in GLSL.
FUNCTION = "pack_overcast"

# The estimate in GLSL: 0 is not overcast, or cannot tell; 1 is rain or thunder.
#
# `fogColour` is the game's fog colour as it stands in the Fog block, not made linear: FOG_COLOUR_SOURCE.
# `skyLight` is the colour of the light map at full sky light and no block light, as sampled, not made
# linear, and not one channel of it: SKY_LIGHT_SOURCE. It is the same for every vertex of a frame.
SIGNATURE = GlslCode(f"float {FUNCTION}(vec3 fogColour, vec3 skyLight)")

# What to pass as `fogColour`. Both terrain.vsh and terrain.fsh are given it, by fog.glsl.
FOG_COLOUR_SOURCE = GlslCode("FogColor.rgb")

# The light map takes a light level in sixteenths of a texel, as terrain.vsh's UV2 holds it.
_LIGHT_STEPS_PER_LEVEL = 16
_FULL_SKY_LIGHT = 15
_NO_BLOCK_LIGHT = 0

# What to pass as `skyLight`. Only terrain.vsh is given the light map, and not where OIT_ALPHA_ONLY is
# defined; terrain.fsh has to be handed the estimate, or the colour, by the vertex stage.
SKY_LIGHT_SOURCE = GlslCode(
    f"sample_lightmap(Sampler2, ivec2({_NO_BLOCK_LIGHT * _LIGHT_STEPS_PER_LEVEL},"
    f" {_FULL_SKY_LIGHT * _LIGHT_STEPS_PER_LEVEL})).rgb"
)

_GUARD = "PACK_WEATHER_GLSL"

# Each line is the line of `overcast` in the same place. Change one and change the other.
_GLSL = ShaderText(f"""#ifndef {_GUARD}
#define {_GUARD}

{SIGNATURE} {{
    float fogWarmth = fogColour.r / max(fogColour.b, {_BLACK});
    float air = clamp(({_AIR_TOLERANCE} - abs(fogWarmth - {_AIR_WARMTH})) / {_AIR_EDGE}, {_NONE}, {_FULL});
    float skyWarmth = skyLight.r / max(skyLight.b, {_BLACK});
    float daylight = clamp((skyWarmth - {_NIGHT_WARMTH}) / ({_DAY_WARMTH} - {_NIGHT_WARMTH}), {_NONE}, {_FULL});
    float overworld = step(skyWarmth, {_WARMEST_SKY});
    float clearFog = min({_FULL}, {_NIGHTFALL_FOG} + daylight);
    float darkening = {_FULL} - fogColour.b / clearFog;
    float cover = clamp((darkening - {_CLEAR_DARKENING}) / ({_RAIN_DARKENING} - {_CLEAR_DARKENING}), {_NONE}, {_FULL});
    float sure = clamp(daylight / {_SURE_DAYLIGHT}, {_NONE}, {_FULL});
    return cover * sure * overworld * air;
}}

#endif
""")


def _is_level(value: float) -> bool:
    return _NONE <= value <= _FULL


def _is_colour(colour: Colour) -> bool:
    return _is_level(colour.red) and _is_level(colour.green) and _is_level(colour.blue)


@deal.pure
@deal.post(_is_level)
@deal.ensure(lambda value, result: result == value or result in (_NONE, _FULL))
def _clamp(value: float) -> Level:
    """GLSL's clamp of a value to 0..1."""
    return Level(min(max(value, _NONE), _FULL))


@deal.pure
@deal.ensure(lambda edge, value, result: result == (_NONE if value < edge else _FULL))
def _step(edge: float, value: float) -> Level:
    """GLSL's step: nothing under the edge, full from the edge on."""
    return _NONE if value < edge else _FULL


@deal.pure
@deal.pre(lambda fog, sky_light: _is_colour(fog) and _is_colour(sky_light))
@deal.post(_is_level)
def overcast(fog: Colour, sky_light: Colour) -> Level:
    """How overcast the sky is: 0 is not overcast, or cannot tell; 1 is rain or thunder.

    `fog` is the game's fog colour. `sky_light` is the colour of the light map at full sky light and no
    block light. Its brightness is not used, only its warmth: the brightness setting of the player scales
    all three channels alike.
    """
    fog_warmth = fog.red / max(fog.blue, _BLACK)
    air = _clamp((_AIR_TOLERANCE - abs(fog_warmth - _AIR_WARMTH)) / _AIR_EDGE)
    sky_warmth = sky_light.red / max(sky_light.blue, _BLACK)
    daylight = _clamp((sky_warmth - _NIGHT_WARMTH) / (_DAY_WARMTH - _NIGHT_WARMTH))
    overworld = _step(sky_warmth, _WARMEST_SKY)
    clear_fog = min(_FULL, _NIGHTFALL_FOG + daylight)
    darkening = _FULL - fog.blue / clear_fog
    cover = _clamp((darkening - _CLEAR_DARKENING) / (_RAIN_DARKENING - _CLEAR_DARKENING))
    sure = _clamp(daylight / _SURE_DAYLIGHT)
    return Level(cover * sure * overworld * air)


@deal.pure
@deal.post(lambda result: result == ShaderFile(INCLUDE_PATH, _GLSL))
def include_file() -> ShaderFile:
    """The include file that holds the estimate, for the pack and for whoever checks a shader that calls it."""
    return ShaderFile(INCLUDE_PATH, _GLSL)


@deal.pure
@deal.post(lambda result: result == Effect(NAME, (), (include_file(),)))
def effect() -> Effect:
    """The weather: it changes no shader, and puts the estimate in the pack for the effects that call it."""
    return Effect(NAME, (), (include_file(),))
