"""The lighting gives every light level a colour: sun against cool shade, fire no bluer than the game's.

The requirements are checked on the Python maths of the effect, since GLSL cannot be run here. What ties
that maths to the game is checked beside it: the GLSL carries the same constants and the same lines, it
sits at the places of the real lightmap.fsh, and the result compiles as the game compiles it.
"""

from dataclasses import dataclass, replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from democritus.composer import compose
from democritus.effects import Position
from democritus.effects.lighting import (
    BLOCK_LIGHT,
    CLAMP,
    DAY_SKY_FACTOR,
    EMBER,
    FLAME,
    KNEE,
    LIGHTMAP,
    MAIN,
    NAME,
    NIGHT_SKY_FACTOR,
    ROLL_OFF,
    SHADE,
    SKY_LIGHT,
    SUN,
    SUN_REACH,
    WHITE,
    WHITE_POINT,
    Colour,
    Intensity,
    LightLevel,
    Share,
    SkyFactor,
    effect,
    fire_tint,
    shoulder,
    sky_tint,
    sun_share,
)
from democritus.validator import validate
from democritus.vanilla import Game, ShaderText

LEVELS = 16
TOP_LEVEL = LEVELS - 1
FULL = LightLevel(1.0)
NO_LIGHT = Colour(Intensity(0.0), Intensity(0.0), Intensity(0.0))
NEUTRAL = Colour(Intensity(1.0), Intensity(1.0), Intensity(1.0))


def level(number: int) -> LightLevel:
    """A light level from 0 to 15, as lightmap.fsh holds it."""
    return LightLevel(number / TOP_LEVEL)


def colour(red: float, green: float, blue: float) -> Colour:
    return Colour(Intensity(red), Intensity(green), Intensity(blue))


def channels(of: Colour) -> tuple[Intensity, Intensity, Intensity]:
    return (of.red, of.green, of.blue)


def plus(one: Colour, other: Colour) -> Colour:
    return colour(one.red + other.red, one.green + other.green, one.blue + other.blue)


def times(one: Colour, other: Colour) -> Colour:
    return colour(one.red * other.red, one.green * other.green, one.blue * other.blue)


def scaled(one: Colour, by: float) -> Colour:
    return colour(one.red * by, one.green * by, one.blue * by)


def hex_colour(code: int) -> Colour:
    """A colour as the game's data files write it, such as 0x7A7AFF."""
    return colour((code >> 16 & 0xFF) / 0xFF, (code >> 8 & 0xFF) / 0xFF, (code & 0xFF) / 0xFF)


# ---- the game's side of lightmap.fsh, as 26.3 has it, so that whole light maps can be worked out ------------------


@dataclass(frozen=True)
class LightmapInfo:
    """What the game tells lightmap.fsh each frame."""

    sky_factor: SkyFactor
    sky_light_color: Colour
    ambient_color: Colour
    block_factor: float = 1.4
    block_light_tint: Colour = hex_colour(0xFFD88C)
    night_vision_factor: float = 0.0
    night_vision_color: Colour = hex_colour(0x999999)
    darkness_scale: float = 0.0
    boss_darkening: float = 0.0
    brightness_factor: float = 0.5


NIGHT_SKY = hex_colour(0x7A7AFF)
OVERWORLD_AMBIENT = hex_colour(0x0A0A0A)
NOON = LightmapInfo(DAY_SKY_FACTOR, NEUTRAL, OVERWORLD_AMBIENT)
MIDNIGHT = LightmapInfo(NIGHT_SKY_FACTOR, NIGHT_SKY, OVERWORLD_AMBIENT)
# Rain moves the sky factor and the sky light colour 5/16 of the way to the night's.
RAIN = LightmapInfo(SkyFactor(0.7625), NEUTRAL.mixed(NIGHT_SKY, Share(0.3125)), OVERWORLD_AMBIENT)
NETHER = LightmapInfo(SkyFactor(0.0), NIGHT_SKY, hex_colour(0x302821))
THE_END = LightmapInfo(SkyFactor(0.0), hex_colour(0xAC60CD), hex_colour(0x3F473F))
NIGHT_VISION = replace(MIDNIGHT, night_vision_factor=1.0)
DARKNESS = replace(NOON, darkness_scale=0.3, brightness_factor=0.0)
BRIGHTEST = replace(MIDNIGHT, brightness_factor=1.0)
BOSS = replace(NOON, boss_darkening=1.0)
SCENES = (NOON, MIDNIGHT, RAIN, NETHER, THE_END, NIGHT_VISION, DARKNESS, BRIGHTEST, BOSS)

BOSS_TINT = colour(0.7, 0.6, 0.6)


def game_brightness(of: LightLevel) -> float:
    return of / (4.0 - 3.0 * of)


def game_block_light(block_level: LightLevel, tint: Colour) -> Colour:
    """The colour the game gives block light, before its brightness."""
    return tint.mixed(NEUTRAL, Share(0.9 * (2.0 * block_level - 1.0) ** 2))


def game_not_gamma(of: Colour) -> Colour:
    strongest = max(channels(of))
    if strongest == 0.0:
        # The game divides by zero here. It never comes to that in a dimension with ambient light.
        return of
    return scaled(of, (1.0 - (1.0 - strongest) ** 4) / strongest)


def light_map_texel(block_level: LightLevel, sky_level: LightLevel, info: LightmapInfo) -> Colour:
    """One texel of the light map with the pack on: the game's steps in order, with the pack's three changes."""
    night_vision = scaled(info.night_vision_color, info.night_vision_factor)
    lit = Colour(*map(max, channels(info.ambient_color), channels(night_vision)))
    sky = times(info.sky_light_color, sky_tint(sky_level, info.sky_factor))
    lit = plus(lit, scaled(sky, game_brightness(sky_level) * info.sky_factor))
    block = times(game_block_light(block_level, info.block_light_tint), fire_tint(block_level))
    lit = plus(lit, scaled(block, game_brightness(block_level) * info.block_factor))
    lit = lit.mixed(times(lit, BOSS_TINT), Share(info.boss_darkening))
    lit = colour(*(channel - info.darkness_scale for channel in channels(lit)))
    lit = Colour(*map(shoulder, channels(lit)))
    return lit.mixed(game_not_gamma(lit), Share(info.brightness_factor))


def assert_colour(actual: Colour, red: float, green: float, blue: float) -> None:
    assert channels(actual) == pytest.approx((red, green, blue), abs=5e-4)


unit = st.floats(0.0, 1.0)
levels = st.integers(0, TOP_LEVEL).map(level)
shaded_levels = st.integers(0, TOP_LEVEL - 1).map(level)
# The end flash adds to the sky factor, so it can pass 1.
sky_factors = st.floats(0.0, 2.0).map(SkyFactor)
colours = st.builds(colour, unit, unit, unit)
infos = st.builds(
    LightmapInfo,
    sky_factor=sky_factors,
    sky_light_color=colours,
    ambient_color=colours,
    block_factor=st.floats(0.0, 2.0),
    block_light_tint=colours,
    night_vision_factor=unit,
    night_vision_color=colours,
    darkness_scale=unit,
    boss_darkening=unit,
    brightness_factor=unit,
)


# ---- colour: every level has a colour the light map can hold -----------------------------------------------------


@given(levels, levels, infos)
def test_every_texel_is_a_colour_the_light_map_can_hold(block: LightLevel, sky: LightLevel, info: LightmapInfo) -> None:
    assert all(0.0 <= channel <= 1.0 for channel in channels(light_map_texel(block, sky, info)))


def test_noon_is_warm_sun_over_cool_shade() -> None:
    assert_colour(light_map_texel(level(0), level(15), NOON), 0.9988, 0.9888, 0.9518)
    assert_colour(light_map_texel(level(0), level(13), NOON), 0.7219, 0.7789, 0.8380)
    assert_colour(light_map_texel(level(0), level(8), NOON), 0.3571, 0.4173, 0.4935)


def test_a_torch_is_warm_white_at_the_flame_and_amber_at_its_edge() -> None:
    assert_colour(light_map_texel(level(14), level(0), NOON), 0.9769, 0.9062, 0.7409)
    assert_colour(light_map_texel(level(8), level(0), NOON), 0.5861, 0.4097, 0.2250)
    assert_colour(light_map_texel(level(2), level(0), NOON), 0.2043, 0.1587, 0.1204)


def test_torchlight_glows_warm_against_a_blue_night() -> None:
    moonlit = light_map_texel(level(0), level(15), MIDNIGHT)
    torchlit = light_map_texel(level(12), level(15), MIDNIGHT)
    assert_colour(moonlit, 0.2139, 0.2447, 0.5165)
    assert moonlit.blue > 2 * moonlit.red
    assert torchlit.warmer_than(NEUTRAL)


@pytest.mark.parametrize("scene", SCENES)
def test_the_dark_of_a_cave_is_the_games_own(scene: LightmapInfo) -> None:
    # Below the knee the shoulder changes nothing, and with no sky and no block light nothing is tinted.
    floor = max(channels(scene.ambient_color), channels(scaled(scene.night_vision_color, scene.night_vision_factor)))
    assert max(floor) <= KNEE
    darkened = Colour(*floor).mixed(times(Colour(*floor), BOSS_TINT), Share(scene.boss_darkening))
    clamped = colour(*(max(channel - scene.darkness_scale, 0.0) for channel in channels(darkened)))
    expected = clamped.mixed(game_not_gamma(clamped), Share(scene.brightness_factor))
    assert channels(light_map_texel(level(0), level(0), scene)) == pytest.approx(channels(expected))


def test_night_vision_and_the_brightness_slider_still_brighten() -> None:
    plain = light_map_texel(level(0), level(0), MIDNIGHT)
    assert min(channels(light_map_texel(level(0), level(0), NIGHT_VISION))) > 5 * max(channels(plain))
    assert min(channels(light_map_texel(level(0), level(0), BRIGHTEST))) > max(channels(plain))


def test_the_darkness_effect_still_takes_light_away() -> None:
    assert light_map_texel(level(0), level(0), DARKNESS) == NO_LIGHT
    # A level whose light is under the knee, so that the shoulder has no part in the difference.
    darkened = light_map_texel(level(0), level(11), DARKNESS)
    undarkened = light_map_texel(level(0), level(11), replace(DARKNESS, darkness_scale=0.0))
    assert_colour(darkened, *(channel - DARKNESS.darkness_scale for channel in channels(undarkened)))


def test_light_under_the_knee_is_left_as_it_is() -> None:
    assert shoulder(Intensity(0.0)) == 0.0
    assert shoulder(Intensity(0.3)) == 0.3
    assert shoulder(KNEE) == KNEE


def test_light_over_the_knee_rolls_off_to_white() -> None:
    assert shoulder(Intensity(0.8)) == pytest.approx(0.775)
    assert shoulder(Intensity(1.0)) == pytest.approx(0.9)
    assert shoulder(Intensity(1.2)) == pytest.approx(0.975)
    assert shoulder(WHITE_POINT) == pytest.approx(WHITE)
    assert shoulder(Intensity(5.0)) == pytest.approx(WHITE)


def test_light_below_none_is_none() -> None:
    assert shoulder(Intensity(-0.3)) == 0.0


def test_the_shoulder_leaves_the_knee_smoothly_and_is_level_at_the_white_point() -> None:
    assert WHITE_POINT == pytest.approx(1.4)
    assert ROLL_OFF == pytest.approx(0.625)
    step = 1e-6
    assert (shoulder(Intensity(KNEE + step)) - shoulder(KNEE)) / step == pytest.approx(1.0, abs=1e-4)
    assert (shoulder(WHITE_POINT) - shoulder(Intensity(WHITE_POINT - step))) / step == pytest.approx(0.0, abs=1e-4)


@given(st.floats(-1.0, 3.0), st.floats(-1.0, 3.0))
def test_more_light_is_never_shown_as_less(one: float, other: float) -> None:
    dimmer, brighter = sorted((one, other))
    assert shoulder(Intensity(dimmer)) <= shoulder(Intensity(brighter))


# ---- shade: less sky is never warmer ----------------------------------------------------------------------------


def sky_light(sky_level: LightLevel, sky_factor: SkyFactor, sky_light_color: Colour) -> Colour:
    """The colour the lighting gives the sky light of a level: the game's colour seen through the tint.

    The brightness of the level scales all three channels alike, so it has no say in how warm the colour is.
    """
    return times(sky_light_color, sky_tint(sky_level, sky_factor))


@given(shaded_levels, sky_factors, colours)
def test_no_sky_level_below_15_is_warmer_than_15(sky: LightLevel, factor: SkyFactor, game_colour: Colour) -> None:
    assert not sky_light(sky, factor, game_colour).warmer_than(sky_light(FULL, factor, game_colour))


@pytest.mark.parametrize("below", range(TOP_LEVEL))
def test_by_day_each_step_out_of_the_sun_is_cooler(below: int) -> None:
    nearer = sky_light(level(below + 1), DAY_SKY_FACTOR, NEUTRAL)
    assert nearer.warmer_than(sky_light(level(below), DAY_SKY_FACTOR, NEUTRAL))


@pytest.mark.parametrize("below", range(TOP_LEVEL))
def test_by_day_the_whole_light_map_shows_shade_cooler_and_dimmer_than_sun(below: int) -> None:
    sunlit = light_map_texel(level(0), FULL, NOON)
    shaded = light_map_texel(level(0), level(below), NOON)
    assert sunlit.warmer_than(shaded)
    assert all(shade < sun for shade, sun in zip(channels(shaded), channels(sunlit), strict=True))


def test_the_open_sky_by_day_gets_the_sun_and_deep_shade_none_of_it() -> None:
    assert sky_tint(FULL, DAY_SKY_FACTOR) == SUN
    assert sky_tint(level(0), DAY_SKY_FACTOR) == SHADE
    assert SUN.warmer_than(NEUTRAL)
    assert NEUTRAL.warmer_than(SHADE)


@given(levels)
def test_by_night_there_is_no_sun_at_any_level(sky: LightLevel) -> None:
    assert sky_tint(sky, NIGHT_SKY_FACTOR) == SHADE
    assert sky_tint(sky, SkyFactor(0.0)) == SHADE


def test_the_sun_falls_off_fast_with_the_sky_level() -> None:
    assert SUN_REACH == 8.0
    assert sun_share(FULL, DAY_SKY_FACTOR) == 1.0
    assert sun_share(level(14), DAY_SKY_FACTOR) == pytest.approx(0.57583, abs=1e-5)
    assert sun_share(level(12), DAY_SKY_FACTOR) == pytest.approx(0.16777, abs=1e-5)
    assert sun_share(level(0), DAY_SKY_FACTOR) == 0.0


def test_weather_and_dusk_take_part_of_the_sun() -> None:
    assert (NIGHT_SKY_FACTOR, DAY_SKY_FACTOR) == (0.24, 1.0)
    assert sun_share(FULL, SkyFactor(0.7625)) == pytest.approx(0.76807, abs=1e-5)
    assert sun_share(FULL, SkyFactor(0.62)) == pytest.approx(0.5)
    assert sun_share(FULL, SkyFactor(0.43)) == pytest.approx(0.15625)
    assert sun_share(level(14), SkyFactor(0.62)) == pytest.approx(0.28792, abs=1e-5)


def test_a_sky_factor_past_a_clear_day_is_a_clear_day() -> None:
    # The end flash adds to the sky factor.
    assert sun_share(FULL, SkyFactor(2.0)) == 1.0


def test_warmer_is_more_red_for_the_blue() -> None:
    assert colour(0.5, 0.2, 0.4).warmer_than(colour(0.5, 0.9, 0.5))
    assert not colour(0.5, 0.9, 0.5).warmer_than(colour(0.5, 0.2, 0.4))
    assert not colour(0.2, 0.3, 0.4).warmer_than(colour(0.4, 0.1, 0.8))
    assert colour(0.3, 0.0, 0.0).warmer_than(colour(0.3, 0.0, 0.1))


def test_a_colour_mixes_toward_another_channel_by_channel() -> None:
    assert_colour(colour(0.0, 0.4, 1.0).mixed(colour(1.0, 0.8, 0.5), Share(0.25)), 0.25, 0.5, 0.875)
    assert colour(0.1, 0.2, 0.3).mixed(colour(0.9, 0.8, 0.7), Share(0.0)) == colour(0.1, 0.2, 0.3)
    assert_colour(colour(0.1, 0.2, 0.3).mixed(colour(0.9, 0.8, 0.7), Share(1.0)), 0.9, 0.8, 0.7)


def test_a_colour_is_within_two_others_when_every_channel_is() -> None:
    weakest, strongest = colour(0.2, 0.3, 0.4), colour(0.6, 0.7, 0.8)
    assert weakest.within(weakest, strongest)
    assert strongest.within(weakest, strongest)
    assert not colour(0.1, 0.5, 0.5).within(weakest, strongest)
    assert not colour(0.7, 0.5, 0.5).within(weakest, strongest)
    assert not colour(0.5, 0.2, 0.5).within(weakest, strongest)
    assert not colour(0.5, 0.8, 0.5).within(weakest, strongest)
    assert not colour(0.5, 0.5, 0.3).within(weakest, strongest)
    assert not colour(0.5, 0.5, 0.9).within(weakest, strongest)


# ---- warm: block light is never bluer than the game's -----------------------------------------------------------


@given(levels, colours)
def test_block_light_is_never_bluer_than_the_games(block: LightLevel, tint: Colour) -> None:
    vanilla = game_block_light(block, tint)
    assert not vanilla.warmer_than(times(vanilla, fire_tint(block)))


@given(st.floats(0.0, 1.0).map(LightLevel))
def test_the_fire_tint_only_takes_green_and_blue_away(block: LightLevel) -> None:
    tint = fire_tint(block)
    assert tint.red == WHITE
    assert 0.0 < tint.blue < tint.green < tint.red


def test_fire_is_amber_at_the_edge_of_its_reach_and_warm_white_at_the_flame() -> None:
    assert fire_tint(level(0)) == EMBER == colour(1.0, 0.62, 0.30)
    assert_colour(fire_tint(FULL), *channels(FLAME))
    assert FLAME == colour(1.0, 0.92, 0.78)
    assert_colour(fire_tint(LightLevel(0.5)), 1.0, 0.77, 0.54)


@pytest.mark.parametrize("below", range(TOP_LEVEL))
def test_fire_is_warmer_the_further_from_the_flame(below: int) -> None:
    assert fire_tint(level(below)).warmer_than(fire_tint(level(below + 1)))


# ---- the GLSL is the same maths, and fits the real game ----------------------------------------------------------

GLSL_CONSTANTS = (
    "const float PACK_NIGHT_SKY_FACTOR = 0.24;\n"
    "const float PACK_DAY_SKY_FACTOR = 1.0;\n"
    "const float PACK_SUN_REACH = 8.0;\n"
    "const vec3 PACK_SUN = vec3(1.3, 1.22, 1.08);\n"
    "const vec3 PACK_SHADE = vec3(0.7, 0.85, 1.04);\n"
    "const vec3 PACK_EMBER = vec3(1.0, 0.62, 0.3);\n"
    "const vec3 PACK_FLAME = vec3(1.0, 0.92, 0.78);\n"
    "const float PACK_KNEE = 0.6;\n"
    "const float PACK_WHITE_POINT = 1.4;\n"
    "const float PACK_ROLL_OFF = 0.625;\n"
)

# The GLSL as the pack ships it. It is written out here a second time on purpose: nothing here can run it,
# so a change to it has to be made in two places, and is then a decision and not a slip.
GLSL_FUNCTIONS = """
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


def code_at(place: str) -> tuple[Position, str]:
    (attachment,) = (found for found in effect().attachments if found.place == place)
    return attachment.position, attachment.code


def test_the_glsl_holds_the_constants_and_the_lines_of_the_python_maths() -> None:
    assert code_at(MAIN) == (Position.BEFORE, GLSL_CONSTANTS + GLSL_FUNCTIONS)


def test_the_games_sky_light_is_kept_and_seen_through_the_sky_tint() -> None:
    assert code_at(SKY_LIGHT) == (
        Position.INSTEAD,
        "    color += lightmapInfo.SkyLightColor * sky_brightness * packSkyTint(sky_level, lightmapInfo.SkyFactor);\n",
    )


def test_the_games_block_light_is_kept_and_seen_through_the_fire_tint() -> None:
    assert code_at(BLOCK_LIGHT) == (Position.AFTER, "    BlockLightColor *= packFireTint(block_level);\n")


def test_the_shoulder_comes_before_the_games_clamp() -> None:
    assert code_at(CLAMP) == (Position.BEFORE, "    color = packShoulder(color);\n")


def test_the_effect_is_the_lighting_of_the_light_map_alone() -> None:
    lighting = effect()
    assert lighting.name == NAME == "lighting"
    assert {attachment.shader for attachment in lighting.attachments} == {LIGHTMAP}
    assert LIGHTMAP == "core/lightmap.fsh"
    assert len(lighting.attachments) == 4
    assert lighting.includes == ()


def built_light_map(game: Game) -> ShaderText:
    (built,) = compose(game.shaders, (effect(),))
    assert built.path == LIGHTMAP
    return built.text


def test_the_code_sits_at_its_places_in_the_real_light_map(game: Game) -> None:
    text = built_light_map(game)
    assert GLSL_FUNCTIONS + "void main() {\n" in text
    assert f"    {BLOCK_LIGHT}\n    BlockLightColor *= packFireTint(block_level);\n" in text
    assert f"    color = packShoulder(color);\n    {CLAMP}\n" in text
    assert SKY_LIGHT not in text
    assert "sky_brightness * packSkyTint(sky_level, lightmapInfo.SkyFactor);\n" in text


def test_the_game_keeps_its_own_steps_around_the_code(game: Game) -> None:
    (vanilla,) = (shader.text for shader in game.shaders if shader.path == LIGHTMAP)
    restored = built_light_map(game)
    for attachment in effect().attachments:
        games_own = f"    {attachment.place}\n" if attachment.position is Position.INSTEAD else ""
        restored = restored.replace(attachment.code, games_own)
    assert restored == vanilla


def test_the_built_light_map_compiles_as_the_game_compiles_it(game: Game) -> None:
    assert validate(compose(game.shaders, (effect(),)), game.shaders) == ()
