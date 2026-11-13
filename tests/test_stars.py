"""The stars give each star a brightness and a colour of its own, from the number of the star."""

from collections import Counter

import pytest
from hypothesis import given
from hypothesis import strategies as st
from zigdag import verifies

from shader.composer import compose
from shader.effects import Position
from shader.effects.stars import (
    BRIGHTEST,
    DRAW_BITS,
    DRAW_STEPS,
    FAINTEST,
    GAME_STAR,
    STEEPNESS,
    TINT_COLOURS,
    TINT_SHARES,
    VERTICES_PER_STAR,
    Gain,
    Level,
    StarColour,
    StarHash,
    StarNumber,
    Tint,
    VertexIndex,
    colour_of,
    effect,
    gain_of,
    star_hash,
    star_of,
    tint_of,
)
from shader.validator import validate
from shader.vanilla import Game, ShaderFile, ShaderPath

VERTEX_SHADER = ShaderPath("core/stars.vsh")
FRAGMENT_SHADER = ShaderPath("core/stars.fsh")

# The game tries this many random points and keeps about half as stars, so no star's number reaches it.
GAME_TRIES = 1500
UINT_MAX = (1 << 32) - 1
LAST_DRAW = DRAW_STEPS - 1
PERCENT = 100

# Every value the low half of a hash can take, and every value the high half can take: small enough to
# count through, so the shares below are exact and not estimates from a sample.
LOW_HALVES = tuple(StarHash(draw) for draw in range(DRAW_STEPS))
HIGH_HALVES = tuple(StarHash(draw << DRAW_BITS) for draw in range(DRAW_STEPS))

# The stars the game can draw.
GAME_STARS = tuple(StarNumber(number) for number in range(GAME_TRIES))

# What a hash and a colour were when the look was chosen. A change to the mixing or to the look moves these.
KNOWN_HASHES = {0: 0, 1: 1753845952, 2: 3507691905, 784: 2325740056, UINT_MAX: 1734902346}
KNOWN_STAR = StarNumber(2)
KNOWN_COLOUR = StarColour(
    Level(0.8003520490395802), Level(0.7443274056068097), Level(0.5602464343277062), Level(0.8003520490395802)
)

WHITE = StarColour(Level(1.0), Level(1.0), Level(1.0), Level(1.0))
TINTED = tuple(tint for tint in Tint if tint is not Tint.WHITE)

# A tint this far from white in some channel can be seen; one with a channel below the floor is a lamp, not a star.
SEEN_STEP = 0.1
SUBTLE_FLOOR = 0.5

HALFWAY = Gain((FAINTEST + BRIGHTEST) / 2)
TENTH_OF_THE_WAY = Gain(FAINTEST + (BRIGHTEST - FAINTEST) / 10)

stars = st.integers(0, UINT_MAX).map(StarNumber)
hashes = st.integers(0, UINT_MAX).map(StarHash)
draws = st.integers(0, LAST_DRAW)
vertices = st.integers(0, UINT_MAX).map(VertexIndex)


def channels(colour: StarColour) -> tuple[Level, Level, Level]:
    return (colour.red, colour.green, colour.blue)


def built(game: Game) -> dict[ShaderPath, ShaderFile]:
    return {file.path: file for file in compose(game.shaders, (effect(),))}


@verifies("shader.effects.stars#vary")
@given(st.integers(0, UINT_MAX // VERTICES_PER_STAR).map(StarNumber))
def test_the_four_corners_of_a_quad_are_one_star_and_the_next_quad_is_the_next_star(star: StarNumber) -> None:
    first = star * VERTICES_PER_STAR
    assert {star_of(VertexIndex(first + corner)) for corner in range(VERTICES_PER_STAR)} == {star}
    assert star_of(VertexIndex(first + VERTICES_PER_STAR)) == star + 1


@verifies("shader.effects.stars#vary")
@given(stars)
def test_the_same_number_always_gives_the_same_star(star: StarNumber) -> None:
    assert colour_of(star) == colour_of(StarNumber(int(star)))


@verifies("shader.effects.stars#vary")
def test_the_stars_are_the_ones_the_look_was_chosen_with() -> None:
    assert {number: star_hash(StarNumber(number)) for number in KNOWN_HASHES} == KNOWN_HASHES
    assert colour_of(KNOWN_STAR) == KNOWN_COLOUR


@verifies("shader.effects.stars#vary")
def test_no_two_stars_of_the_game_are_given_the_same_hash() -> None:
    assert len({star_hash(star) for star in GAME_STARS}) == GAME_TRIES


@verifies("shader.effects.stars#vary")
def test_nearly_every_star_of_the_game_has_a_colour_no_other_star_has() -> None:
    # Measured: 1500 of 1500. The floor leaves room for a new look to make a few stars alike, not many.
    assert len({colour_of(star) for star in GAME_STARS}) >= GAME_TRIES * 99 // PERCENT


@verifies("shader.effects.stars#vary")
def test_stars_next_to_each_other_in_number_are_no_more_alike_than_any_two() -> None:
    # A weak hash gives neighbours in number the same tint in runs. With these shares two unrelated stars
    # share a tint 44 times in 100, so 50 leaves room for chance and none for runs.
    alike = sum(tint_of(star_hash(a)) is tint_of(star_hash(b)) for a, b in zip(GAME_STARS, GAME_STARS[1:]))
    assert alike * PERCENT < 50 * (GAME_TRIES - 1)


@verifies("shader.effects.stars#vary")
@given(stars)
def test_every_channel_stays_between_nothing_and_the_brightest_star(star: StarNumber) -> None:
    colour = colour_of(star)
    assert FAINTEST <= colour.alpha < BRIGHTEST
    assert all(0.0 < level <= colour.alpha for level in channels(colour))
    assert max(channels(colour)) == colour.alpha


@verifies("shader.effects.stars#vary")
@given(stars)
def test_a_star_is_its_tint_at_its_gain(star: StarNumber) -> None:
    mixed = star_hash(star)
    gain = gain_of(mixed)
    tint = TINT_COLOURS[tint_of(mixed)]
    assert colour_of(star) == StarColour(
        Level(tint.red * gain), Level(tint.green * gain), Level(tint.blue * gain), Level(gain)
    )


@verifies("shader.effects.stars#vary")
@given(draws, draws, draws)
def test_gain_and_tint_are_drawn_from_different_halves_of_the_hash(low: int, high: int, other: int) -> None:
    mixed = StarHash(high << DRAW_BITS | low)
    assert gain_of(mixed) == gain_of(StarHash(other << DRAW_BITS | low))
    assert tint_of(mixed) is tint_of(StarHash(high << DRAW_BITS | other))


@verifies("shader.effects.stars#vary")
def test_the_gain_runs_from_the_faintest_star_to_just_under_the_brightest() -> None:
    gains = [gain_of(mixed) for mixed in LOW_HALVES]
    assert gains[0] == FAINTEST
    assert gains == sorted(gains)
    assert gains[LAST_DRAW] == pytest.approx(BRIGHTEST, rel=1e-3)
    assert gains[DRAW_STEPS // 2] == FAINTEST + (BRIGHTEST - FAINTEST) * 0.5**STEEPNESS


@verifies("shader.effects.stars#vary")
def test_most_stars_are_faint_and_a_few_are_bright() -> None:
    gains = [gain_of(mixed) for mixed in LOW_HALVES]
    within_a_tenth = sum(gain <= TENTH_OF_THE_WAY for gain in gains)
    past_halfway = sum(gain >= HALFWAY for gain in gains)
    brighter_than_the_game = sum(gain > GAME_STAR for gain in gains)
    # 0.1 ** (1 / 4) of the draws stay within a tenth of the way up.
    assert within_a_tenth * PERCENT // DRAW_STEPS == 56
    # 1 - 0.5 ** (1 / 4) of them pass halfway.
    assert past_halfway * PERCENT // DRAW_STEPS == 15
    # The game's own star sits a sixth of the way up: the sky is not dimmer for the change.
    assert brighter_than_the_game * PERCENT // DRAW_STEPS == 36


@verifies("shader.effects.stars#vary")
def test_each_tint_has_its_share_of_the_stars() -> None:
    counted = Counter(tint_of(mixed) for mixed in HIGH_HALVES)
    for tint in Tint:
        # A share is a whole number of draws, so it can be one draw off the exact percentage.
        assert abs(counted[tint] * PERCENT - TINT_SHARES[tint] * DRAW_STEPS) <= PERCENT
    assert sum(TINT_SHARES.values()) == PERCENT


@verifies("shader.effects.stars#vary")
def test_most_stars_are_white_and_a_minority_is_tinted() -> None:
    assert TINT_SHARES[Tint.WHITE] == 62
    assert [TINT_SHARES[tint] for tint in TINTED] == [17, 12, 6, 3]
    assert tint_of(StarHash(0)) is Tint.WHITE
    assert tint_of(StarHash(UINT_MAX)) is Tint.RED


@verifies("shader.effects.stars#vary")
def test_the_game_s_own_stars_are_mostly_white_with_every_tint_among_them() -> None:
    counted = Counter(tint_of(star_hash(star)) for star in GAME_STARS)
    for tint in Tint:
        # Three percentage points is what chance moves a share by among 1500 stars, with room to spare.
        assert abs(counted[tint] * PERCENT - TINT_SHARES[tint] * GAME_TRIES) <= 3 * GAME_TRIES


@verifies("shader.effects.stars#vary")
def test_a_tint_can_be_seen_and_is_still_a_star() -> None:
    assert TINT_COLOURS[Tint.WHITE] == WHITE
    for tint in TINTED:
        colour = TINT_COLOURS[tint]
        assert max(channels(colour)) == 1.0 == colour.alpha
        assert SUBTLE_FLOOR <= min(channels(colour)) <= 1.0 - SEEN_STEP


@verifies("shader.effects.stars#vary")
def test_hot_stars_lean_blue_and_cool_stars_lean_red() -> None:
    blue_white = TINT_COLOURS[Tint.BLUE_WHITE]
    assert blue_white.red < blue_white.green < blue_white.blue
    warm = [TINT_COLOURS[tint] for tint in (Tint.PALE_YELLOW, Tint.ORANGE, Tint.RED)]
    assert all(colour.red > colour.green > colour.blue for colour in warm)
    # Cooler is redder: less green from yellow to orange to red.
    assert [colour.green for colour in warm] == sorted((colour.green for colour in warm), reverse=True)


@verifies("shader.effects.stars#vary")
def test_scaling_a_colour_scales_every_channel() -> None:
    assert StarColour(Level(1.0), Level(0.5), Level(0.25), Level(1.0)).scaled(Gain(2.0)) == StarColour(
        Level(2.0), Level(1.0), Level(0.5), Level(2.0)
    )


@verifies("shader.effects.stars#vary")
def test_the_effect_attaches_to_the_two_star_shaders_and_adds_no_file() -> None:
    stars_effect = effect()
    assert stars_effect.name == "stars"
    assert stars_effect.includes == ()
    assert [(a.shader, a.position) for a in stars_effect.attachments] == [
        (VERTEX_SHADER, Position.BEFORE),
        (VERTEX_SHADER, Position.AFTER),
        (FRAGMENT_SHADER, Position.BEFORE),
        (FRAGMENT_SHADER, Position.INSTEAD),
    ]


@verifies("shader.effects.stars#vary")
def test_the_effect_joins_onto_the_game_s_star_shaders(game: Game) -> None:
    assert set(built(game)) == {VERTEX_SHADER, FRAGMENT_SHADER}


@verifies("shader.effects.stars#vary")
def test_the_built_star_shaders_compile_as_the_game_compiles_them(game: Game) -> None:
    assert validate(tuple(built(game).values()), game.shaders) == ()


@verifies("shader.effects.stars#vary")
def test_the_vertex_stage_numbers_the_stars_and_works_out_the_same_maths(game: Game) -> None:
    text = built(game)[VERTEX_SHADER].text
    assert f"starHash(uint(gl_VertexIndex / {VERTICES_PER_STAR}));" in text
    assert f"float draw = float(mixed & {LAST_DRAW}u) / {float(DRAW_STEPS)};" in text
    assert f"return {FAINTEST} + {BRIGHTEST - FAINTEST} * pow(draw, {STEEPNESS});" in text
    assert f"uint draw = mixed >> {DRAW_BITS}u;" in text
    for colour in TINT_COLOURS.values():
        assert f"vec4({colour.red}, {colour.green}, {colour.blue}, {colour.alpha})" in text
    # The game's own line still places the corner: the effect adds to it and takes nothing away.
    assert "gl_Position = ProjMat * ModelViewMat * vec4(Position, 1.0);" in text


@verifies("shader.effects.stars#vary")
def test_the_fragment_stage_keeps_the_game_s_fade(game: Game) -> None:
    lines = [line.strip() for line in built(game)[FRAGMENT_SHADER].text.splitlines()]
    assert "fragColor = starColour * ColorModulator;" in lines
    assert "fragColor = ColorModulator;" not in lines
    assert "layout(location = 0) flat in vec4 starColour;" in lines
    assert "layout(location = 0) flat out vec4 starColour;" in built(game)[VERTEX_SHADER].text
