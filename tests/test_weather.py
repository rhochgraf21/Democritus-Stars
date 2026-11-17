"""The weather estimate against the game's own numbers, and its include file against the game's own shaders."""

from types import SimpleNamespace

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from democritus.composer import compose
from democritus.effects import Attachment, Effect, EffectName, ExactLine, GlslCode, Place, Position
from democritus.effects import weather
from democritus.effects.weather import Colour, Level, overcast
from democritus.validator import validate
from democritus.vanilla import Game, ShaderPath

EIGHT_BIT = 255
LOW = 0.02  # an estimate under this thickens no fog that anyone could see
ROUNDING = 1e-9


def colour(red: int, green: int, blue: int) -> Colour:
    return Colour(Level(red / EIGHT_BIT), Level(green / EIGHT_BIT), Level(blue / EIGHT_BIT))


# What 26.3 sends, worked out from its WeatherAttributes, day timeline, AtmosphericFogEnvironment and
# lightmap.fsh: the fog colour at a render distance of 12 chunks, and the light map at full sky light with
# the brightness setting at its default of 50 %. Ticks count from sunrise; noon is 6000, midnight 18000.
NOON_LIGHT = colour(255, 255, 255)
RAIN_LIGHT = colour(194, 194, 230)
THUNDER_LIGHT = colour(153, 153, 207)
NIGHT_LIGHT = colour(71, 71, 129)

PLAINS_NOON = colour(181, 209, 255)
PLAINS_RAIN = colour(89, 100, 143)
PLAINS_THUNDER = colour(42, 47, 67)

CLEAR = {
    "noon in the plains": (PLAINS_NOON, NOON_LIGHT),
    "noon in the desert": (colour(180, 210, 255), NOON_LIGHT),
    "noon on the frozen peaks": (colour(183, 207, 255), NOON_LIGHT),
    "noon in the dappled forest": (colour(192, 208, 230), NOON_LIGHT),
    "dusk at tick 12000": (colour(169, 194, 238), colour(179, 179, 222)),
    "dusk at tick 12500": (colour(121, 139, 172), colour(130, 130, 191)),
    "dusk at tick 13000": (colour(73, 84, 107), colour(84, 84, 146)),
    "dusk at tick 12500 at the darkest brightness setting": (colour(121, 139, 172), colour(94, 94, 138)),
    "dusk at tick 12500 at the brightest brightness setting": (colour(121, 139, 172), colour(166, 166, 244)),
    "midnight": (colour(10, 11, 19), NIGHT_LIGHT),
}

WET = {
    "rain in the plains": (PLAINS_RAIN, RAIN_LIGHT),
    "rain in the desert": (colour(89, 101, 143), RAIN_LIGHT),
    "rain in the dappled forest": (colour(94, 100, 128), RAIN_LIGHT),
    "rain at a render distance of 32 chunks": (colour(96, 108, 153), RAIN_LIGHT),
    "rain at the darkest brightness setting": (PLAINS_RAIN, colour(173, 173, 204)),
    "rain at the brightest brightness setting": (PLAINS_RAIN, colour(215, 215, 255)),
    "thunder in the plains": (PLAINS_THUNDER, THUNDER_LIGHT),
    "thunder in the dappled forest": (colour(45, 47, 60), THUNDER_LIGHT),
    "rain at dusk, tick 12000": (colour(83, 93, 133), colour(144, 144, 200)),
    "rain at dusk, tick 12500": (colour(60, 67, 97), colour(111, 111, 175)),
}

# Where the two values cannot tell. The fog is as dark as rain's, or darker, under a sky light that says day.
CANNOT_TELL = {
    "rain at midnight": (colour(5, 5, 11), NIGHT_LIGHT),
    "under water": (colour(5, 5, 51), NOON_LIGHT),
    "under lava": (colour(153, 26, 0), NOON_LIGHT),
    "blinded": (colour(0, 0, 0), NOON_LIGHT),
    "the End": (colour(24, 19, 24), colour(63, 71, 63)),
    "the Nether's wastes": (colour(51, 8, 8), colour(48, 40, 33)),
    "the Nether's basalt deltas": (colour(104, 95, 112), colour(48, 40, 33)),
    "between the Nether's wastes and its soul sand valley": (colour(35, 49, 48), colour(48, 40, 33)),
    "rain in the pale garden": (colour(64, 60, 69), RAIN_LIGHT),
}

# Between clear and rain, and where the estimate fades out. The fog and the light are the game's; the
# estimate beside each is what the constants make of them, kept so that no constant moves unnoticed.
PARTLY = {
    "rain a quarter set in at noon": (colour(158, 181, 226), colour(243, 243, 252), 0.0),
    "rain half set in at noon": (colour(135, 154, 197), colour(226, 226, 245), 0.6454),
    "rain as night falls, tick 13000": (colour(36, 41, 60), colour(80, 80, 141), 0.1118),
    "thunder at dusk, tick 12500": (colour(28, 32, 46), colour(98, 98, 162), 0.6809),
    "rain, turned toward a sunset that warms the fog": (colour(119, 110, 143), RAIN_LIGHT, 0.5566),
    "rain, in a fog bluer than the air's": (colour(72, 90, 143), RAIN_LIGHT, 0.4699),
}


def cases(table: dict) -> list:
    return [pytest.param(*values, id=name) for name, values in table.items()]


@pytest.mark.parametrize(("fog", "sky_light"), cases(CLEAR))
def test_clear_weather_is_not_overcast_at_any_hour(fog: Colour, sky_light: Colour) -> None:
    assert overcast(fog, sky_light) <= LOW


@pytest.mark.parametrize(("fog", "sky_light"), cases(WET))
def test_rain_and_thunder_are_overcast_while_there_is_daylight(fog: Colour, sky_light: Colour) -> None:
    assert overcast(fog, sky_light) == pytest.approx(1.0)


@pytest.mark.parametrize(("fog", "sky_light"), cases(CANNOT_TELL))
def test_where_it_cannot_tell_it_says_not_overcast(fog: Colour, sky_light: Colour) -> None:
    assert overcast(fog, sky_light) == 0.0


@pytest.mark.parametrize(("fog", "sky_light", "estimate"), cases(PARTLY))
def test_between_clear_and_rain_it_is_in_between(fog: Colour, sky_light: Colour, estimate: float) -> None:
    assert overcast(fog, sky_light) == pytest.approx(estimate, abs=5e-5)


levels = st.floats(min_value=0.0, max_value=1.0).map(Level)
colours = st.builds(Colour, levels, levels, levels)
# The fog of the overworld's air and the light of its sky, so that a generated case is seldom one where
# every factor is 0.
air_fogs = st.builds(
    lambda blue, warmth: Colour(Level(blue * warmth), Level(blue), Level(blue)),
    st.floats(min_value=1 / EIGHT_BIT, max_value=1.0),
    st.floats(min_value=0.45, max_value=0.9),
)
sky_lights = st.builds(
    lambda blue, warmth: Colour(Level(blue * warmth), Level(blue * warmth), Level(blue)),
    st.floats(min_value=1 / EIGHT_BIT, max_value=1.0),
    st.floats(min_value=0.5, max_value=1.0),
)


@given(fog=colours, sky_light=colours)
def test_the_estimate_is_always_a_level(fog: Colour, sky_light: Colour) -> None:
    assert 0.0 <= overcast(fog, sky_light) <= 1.0


@given(fog=air_fogs, sky_light=sky_lights, darker=st.floats(min_value=0.0, max_value=1.0))
def test_a_darker_fog_of_the_same_tint_is_never_less_overcast(fog: Colour, sky_light: Colour, darker: float) -> None:
    # Rain does not grey the fog, it darkens it. Under one step of an 8-bit channel a fog counts as black,
    # and black has no tint to keep.
    scale = max(darker, 1 / EIGHT_BIT / fog.blue)
    darkened = Colour(Level(fog.red * scale), Level(fog.green * scale), Level(fog.blue * scale))
    assert overcast(darkened, sky_light) >= overcast(fog, sky_light) - ROUNDING


def glsl_estimate(fog: Colour, sky_light: Colour) -> float:
    """The include file's function, run as Python.

    Its body is nothing but `float name = expression;` lines and a `return`, in arithmetic that Python
    writes the same way, so each line can be handed to Python once GLSL's own functions are given.
    """
    names: dict[str, object] = {
        "clamp": lambda value, low, high: min(max(value, low), high),
        "step": lambda edge, value: 0.0 if value < edge else 1.0,
        "abs": abs,
        "min": min,
        "max": max,
        "fogColour": SimpleNamespace(r=fog.red, g=fog.green, b=fog.blue),
        "skyLight": SimpleNamespace(r=sky_light.red, g=sky_light.green, b=sky_light.blue),
    }
    text = weather.include_file().text
    body = text[text.index(weather.SIGNATURE) :].splitlines()[1:]
    for line in body:
        statement = line.strip().removesuffix(";")
        if statement.startswith("return "):
            return eval(statement.removeprefix("return "), names)
        name, expression = statement.removeprefix("float ").split(" = ", 1)
        names[name] = eval(expression, names)
    raise AssertionError("the function returns nothing")


WARMEST_LIGHT = Colour(Level(0.6), Level(0.6), Level(0.5))  # exactly as warm as a sky light may be


@given(fog=air_fogs | colours, sky_light=sky_lights | colours)
@example(fog=PLAINS_RAIN, sky_light=RAIN_LIGHT)
@example(fog=PLAINS_RAIN, sky_light=WARMEST_LIGHT)
@example(fog=colour(119, 110, 143), sky_light=RAIN_LIGHT)
def test_the_glsl_estimate_is_the_python_estimate(fog: Colour, sky_light: Colour) -> None:
    assert glsl_estimate(fog, sky_light) == pytest.approx(overcast(fog, sky_light), abs=ROUNDING)


def test_a_sky_light_as_warm_as_the_limit_is_still_the_overworlds() -> None:
    assert overcast(PLAINS_RAIN, WARMEST_LIGHT) == pytest.approx(1.0)


PROBE = EffectName("probe")
TERRAIN_FRAGMENT = ShaderPath("core/terrain.fsh")
TERRAIN_VERTEX = ShaderPath("core/terrain.vsh")
MAIN = ExactLine("void main() {")
FRAGMENT_COLOUR_SET = ExactLine("fragColor = calculateFinalColor(color);")
VERTEX_TEXTURE_SET = ExactLine("texCoord0 = UV0;")


def probe(shader: ShaderPath, place: Place, use: str) -> Effect:
    """An effect that calls the estimate once in one shader, as the fog will."""
    return Effect(
        PROBE,
        (
            Attachment(shader, MAIN, Position.BEFORE, weather.INCLUDE_LINE),
            Attachment(shader, place, Position.AFTER, GlslCode(use)),
        ),
    )


def test_the_estimate_compiles_in_the_games_terrain_fragment_shader(game: Game) -> None:
    # The fragment stage is not given the light map, so the vertex colour stands in for the sky light.
    use = f"fragColor.a *= {weather.FUNCTION}({weather.FOG_COLOUR_SOURCE}, vertexColor.rgb);"
    built = compose(game.shaders, (weather.effect(), probe(TERRAIN_FRAGMENT, FRAGMENT_COLOUR_SET, use)))
    assert weather.include_file() in built
    assert validate(built, game.shaders) == ()


def test_the_estimate_compiles_from_the_values_the_terrain_vertex_shader_is_given(game: Game) -> None:
    use = (
        "#ifndef OIT_ALPHA_ONLY\n"
        f"float probed = {weather.FUNCTION}({weather.FOG_COLOUR_SOURCE}, {weather.SKY_LIGHT_SOURCE});\n"
        "#endif"
    )
    built = compose(game.shaders, (weather.effect(), probe(TERRAIN_VERTEX, VERTEX_TEXTURE_SET, use)))
    # sample_lightmap.glsl takes sixteenths of a texel: block light 0, sky light 15 of 15.
    assert weather.SKY_LIGHT_SOURCE == "sample_lightmap(Sampler2, ivec2(0, 240)).rgb"
    assert validate(built, game.shaders) == ()


def test_the_effect_changes_no_shader_and_adds_the_include() -> None:
    made = weather.effect()
    assert made.attachments == ()
    assert [file.path for file in made.includes] == ["include/pack_weather.glsl"]
    assert weather.INCLUDE_LINE == "#include <minecraft:pack_weather.glsl>"
    assert made.includes[0].text.count(weather.SIGNATURE) == 1
