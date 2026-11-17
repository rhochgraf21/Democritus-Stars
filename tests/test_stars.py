"""The stars are the brightest stars of the real sky, with the lines of the constellations, and they twinkle."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from democritus.composer import compose
from democritus.effects.stars import (
    BOX_COUNT,
    BOX_FACES,
    BRIGHTEST,
    FAINTEST,
    GAME_STAR,
    GLOBALS_INCLUDE,
    LARGEST,
    LINE_COUNT,
    LINE_GAP,
    LINE_HALF_WIDTH,
    NAME,
    QUAD_COUNT,
    SMALLEST,
    STAR_COUNT,
    STAR_DISTANCE,
    TINT_COLOURS,
    TWINKLE_DEPTH,
    VERTICES_PER_STAR,
    Clock,
    CornerNumber,
    Direction,
    StarNumber,
    Tint,
    VertexIndex,
    box_corner,
    dot_strength,
    brightness,
    colour_of,
    direction_of,
    drawn_stars,
    effect,
    gain_of,
    galactic,
    line_corner,
    size_of,
    star_corner,
    star_hash,
    star_of,
    tint_of,
    twinkle,
)
from democritus.effects.stars import catalogue, figures, galaxy
from democritus.effects.stars.catalogue import ColourIndex, Degrees, Hours, Magnitude
from democritus.effects.stars.figures import SkyPlace
from democritus.effects.stars.galaxy import GalacticDirection, Glow, Reach
from democritus.validator import validate
from democritus.vanilla import Game, ShaderFile

VERTEX_SHADER = "core/stars.vsh"
FRAGMENT_SHADER = "core/stars.fsh"

# How far two lengths may differ and still be the same: the sums round in the last bits.
CLOSE = 1e-9

# The four corners of a quad, in the order the game draws them.
CORNERS = tuple(CornerNumber(corner) for corner in range(VERTICES_PER_STAR))

# Stars known to anyone who has looked up, as the catalogue should hold them: hours, degrees, magnitude.
SIRIUS = (6.752, -16.716, -1.46)
VEGA = (18.616, 38.784, 0.03)
POLARIS = (2.530, 89.264, 2.02)

clocks = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
star_numbers = st.integers(min_value=0, max_value=STAR_COUNT - 1)
sky_places = st.builds(
    SkyPlace,
    st.floats(min_value=0.0, max_value=23.999, allow_nan=False),
    st.floats(min_value=-90.0, max_value=90.0, allow_nan=False),
)


def text_of(files: tuple[ShaderFile, ...], path: str) -> str:
    return next(file.text for file in files if file.path == path)


def dot(a: Direction, b: Direction) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def minus(a: Direction, b: Direction) -> Direction:
    return Direction(a.x - b.x, a.y - b.y, a.z - b.z)


def cross(a: Direction, b: Direction) -> Direction:
    return Direction(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def faces_the_camera(corners: list[Direction]) -> bool:
    """Whether going round a quad's corners is anticlockwise seen from the camera, which is how the game draws."""
    normal = cross(minus(corners[1], corners[0]), minus(corners[2], corners[1]))
    return dot(normal, corners[0]) < 0


def find(hours: float, degrees: float) -> catalogue.CatalogueStar:
    return min(
        catalogue.stars(),
        key=lambda star: abs(star.right_ascension - hours) + abs(star.declination - degrees),
    )


# ---- catalogue: the brightest stars of the real sky --------------------------------------------------------------


def test_the_catalogue_holds_a_star_for_every_quad_the_game_draws_brightest_first() -> None:
    stars = catalogue.stars()
    assert len(stars) == QUAD_COUNT
    assert [star.magnitude for star in stars] == sorted(star.magnitude for star in stars)


@pytest.mark.parametrize("known", (SIRIUS, VEGA, POLARIS))
def test_the_catalogue_holds_well_known_stars_where_and_as_bright_as_they_are(known: tuple[float, float, float]) -> None:
    hours, degrees, magnitude = known
    star = find(hours, degrees)
    assert star.right_ascension == pytest.approx(hours, abs=0.01)
    assert star.declination == pytest.approx(degrees, abs=0.01)
    assert star.magnitude == pytest.approx(magnitude, abs=0.01)


# ---- figures: the lines of the constellations -----------------------------------------------------------------------


def test_the_figures_leave_more_quads_for_stars_than_they_take_for_lines() -> None:
    assert 0 < LINE_COUNT == len(figures.lines()) < STAR_COUNT
    assert BOX_COUNT + LINE_COUNT + STAR_COUNT == QUAD_COUNT


def test_the_southern_cross_is_two_lines_between_its_four_stars() -> None:
    # Acrux to Gacrux and Mimosa to Delta Crucis, to a tenth of a degree.
    acrux, gacrux = SkyPlace(Hours(12.443), Degrees(-63.10)), SkyPlace(Hours(12.519), Degrees(-57.11))
    near = [
        line for line in figures.lines()
        if abs(line.start.right_ascension - acrux.right_ascension) < 0.01
        and abs(line.start.declination - acrux.declination) < 0.1
    ]
    assert any(abs(line.end.declination - gacrux.declination) < 0.1 for line in near)


def test_most_lines_end_at_a_star_that_is_drawn() -> None:
    drawn = [direction_of(SkyPlace(s.right_ascension, s.declination)) for s in drawn_stars()]
    half_a_degree = math.cos(math.radians(0.5))
    ends = [direction_of(place) for line in figures.lines() for place in (line.start, line.end)]
    at_a_star = sum(any(dot(end, star) > half_a_degree for star in drawn) for end in ends)
    # Measured: 88 %. A line's end without a star is a star too faint for the catalogue.
    assert at_a_star * 100 >= len(ends) * 85


# ---- real: each star at its place, brightness and colour ----------------------------------------------------------


def test_the_real_sky_s_north_pole_lies_on_the_axis_the_game_turns_the_stars_about() -> None:
    pole = direction_of(SkyPlace(Hours(0.0), Degrees(90.0)))
    assert (pole.x, pole.y, pole.z) == pytest.approx((-1.0, 0.0, 0.0), abs=CLOSE)
    polaris = direction_of(SkyPlace(Hours(POLARIS[0]), Degrees(POLARIS[1])))
    assert dot(polaris, pole) > math.cos(math.radians(1.0))


@given(sky_places, sky_places, sky_places)
def test_the_sky_is_turned_and_not_mirrored(one: SkyPlace, two: SkyPlace, three: SkyPlace) -> None:
    def real(place: SkyPlace) -> Direction:
        around, above = math.radians(place.right_ascension * 15), math.radians(place.declination)
        return Direction(math.cos(above) * math.cos(around), math.cos(above) * math.sin(around), math.sin(above))

    # Three directions keep their handedness, and the angles between them, under a rotation and no other map.
    handed = dot(cross(real(one), real(two)), real(three))
    drawn = dot(cross(direction_of(one), direction_of(two)), direction_of(three))
    assert drawn == pytest.approx(handed, abs=1e-6)
    assert dot(direction_of(one), direction_of(two)) == pytest.approx(dot(real(one), real(two)), abs=1e-6)


def test_a_brighter_star_is_drawn_brighter_and_larger() -> None:
    stars = catalogue.stars()
    assert gain_of(stars[0].magnitude) == BRIGHTEST and size_of(stars[0].magnitude) == LARGEST
    assert gain_of(stars[-1].magnitude) == FAINTEST and size_of(stars[-1].magnitude) == SMALLEST
    gains = [gain_of(star.magnitude) for star in stars]
    assert gains == sorted(gains, reverse=True)
    assert stars[0] in drawn_stars()
    assert brightness(Magnitude(99.0)) == 0.0


def test_a_star_has_the_tint_of_its_colour_index() -> None:
    assert tint_of(ColourIndex(-0.2)) is Tint.BLUE_WHITE
    assert tint_of(ColourIndex(0.0)) is Tint.WHITE
    assert tint_of(ColourIndex(0.65)) is Tint.PALE_YELLOW
    assert tint_of(ColourIndex(1.2)) is Tint.ORANGE
    assert tint_of(ColourIndex(1.85)) is Tint.RED
    betelgeuse = find(5.92, 7.41)
    assert colour_of(betelgeuse).blue < colour_of(betelgeuse).red
    assert all(max(c.red, c.green, c.blue) <= 1.0 for c in TINT_COLOURS.values())


@given(sky_places, st.floats(min_value=0.05, max_value=1.0))
def test_a_star_s_quad_is_a_square_about_its_direction_that_faces_the_camera(place: SkyPlace, half_side: float) -> None:
    toward = direction_of(place)
    corners = [star_corner(toward, half_side, corner) for corner in CORNERS]
    assert faces_the_camera(corners)
    centre = Direction(*(sum(getattr(c, axis) for c in corners) / 4 for axis in "xyz"))
    assert (centre.x, centre.y, centre.z) == pytest.approx(
        (toward.x * STAR_DISTANCE, toward.y * STAR_DISTANCE, toward.z * STAR_DISTANCE), abs=1e-6
    )
    side = minus(corners[1], corners[0])
    assert math.sqrt(dot(side, side)) == pytest.approx(2 * half_side, abs=1e-6)


@given(st.integers(min_value=0, max_value=QUAD_COUNT * VERTICES_PER_STAR - 1))
def test_the_four_corners_of_a_quad_belong_to_one_star(vertex: int) -> None:
    star = star_of(VertexIndex(vertex))
    assert star_of(VertexIndex(star * VERTICES_PER_STAR)) == star
    assert star_of(VertexIndex(star * VERTICES_PER_STAR + VERTICES_PER_STAR - 1)) == star


# ---- lines: the figures are drawn ------------------------------------------------------------------------------------


def test_every_line_s_quad_faces_the_camera_and_stops_short_of_both_its_ends() -> None:
    for line in figures.lines():
        start, end = direction_of(line.start), direction_of(line.end)
        corners = [line_corner(start, end, corner) for corner in CORNERS]
        assert faces_the_camera(corners)
        length = math.sqrt(dot(minus(end, start), minus(end, start))) * STAR_DISTANCE
        long_side = minus(corners[1], corners[0])
        assert math.sqrt(dot(long_side, long_side)) == pytest.approx(length * (1 - 2 * LINE_GAP), abs=1e-6)
        short_side = minus(corners[2], corners[1])
        assert math.sqrt(dot(short_side, short_side)) == pytest.approx(2 * LINE_HALF_WIDTH, abs=1e-6)


def test_the_vertex_stage_draws_the_lines_first_and_the_stars_after(game: Game) -> None:
    vertex = text_of(compose(game.shaders, (effect(),)), VERTEX_SHADER)
    assert f"const int PACK_LINE_COUNT = {LINE_COUNT};" in vertex
    assert f"const int PACK_STAR_COUNT = {STAR_COUNT};" in vertex
    assert vertex.count("    vec3(") == 2 * LINE_COUNT + BOX_COUNT
    assert "gl_Position = ProjMat * ModelViewMat * vec4(Position, 1.0);" not in vertex


# ---- twinkle and wrap ------------------------------------------------------------------------------------------------


@given(star_numbers, clocks)
def test_a_twinkle_dims_a_star_by_no_more_than_its_depth_and_never_brightens_it(star: int, clock: float) -> None:
    assert GAME_STAR - TWINKLE_DEPTH <= twinkle(StarNumber(star), Clock(clock)) <= GAME_STAR


def test_a_star_s_brightness_changes_within_a_second_and_no_two_stars_keep_time() -> None:
    # A second is a twelve-hundredth of the clock.
    second = 1.0 / 1200
    over_a_second = [twinkle(StarNumber(3), Clock(step * second / 20)) for step in range(21)]
    assert max(over_a_second) - min(over_a_second) > TWINKLE_DEPTH / 4
    at_once = {round(twinkle(StarNumber(star), Clock(0.3)), 3) for star in range(50)}
    assert len(at_once) > 25
    assert len({star_hash(StarNumber(star)) for star in range(STAR_COUNT)}) == STAR_COUNT


@given(star_numbers)
def test_a_star_is_as_bright_when_the_clock_starts_again_as_when_it_ended(star: int) -> None:
    assert twinkle(StarNumber(star), Clock(1.0)) == pytest.approx(twinkle(StarNumber(star), Clock(0.0)), abs=1e-6)


def test_the_vertex_stage_asks_the_game_for_the_clock(game: Game) -> None:
    vertex = text_of(compose(game.shaders, (effect(),)), VERTEX_SHADER)
    assert vertex.index(GLOBALS_INCLUDE) < vertex.index("void main()")
    assert "packTwinkle(uint(packStar), GameTime)" in vertex


# ---- galaxy: the Milky Way ---------------------------------------------------------------------------------------------


def toward(longitude: float, latitude: float) -> GalacticDirection:
    """The direction at a galactic longitude and latitude, in degrees."""
    around, above = math.radians(longitude), math.radians(latitude)
    return GalacticDirection(math.cos(above) * math.cos(around), math.cos(above) * math.sin(around), math.sin(above))


def brightness_of(glow: Glow) -> float:
    return glow.red + glow.green + glow.blue


# How much of the sky a pixel covers on a screen 1080 pixels high that shows 70 degrees from top to bottom.
PIXEL = Reach(math.radians(70.0) / 1080)


def light(direction: GalacticDirection) -> Glow:
    return galaxy.light_toward(direction, PIXEL)


def mean_glow(longitudes: range, latitude: float) -> Glow:
    """The light averaged along a stretch of one latitude, which evens out the star clouds and the dust."""
    glows = [light(toward(longitude, latitude)) for longitude in longitudes]
    return Glow(*(math.fsum(getattr(glow, colour) for glow in glows) / len(glows) for colour in ("red", "green", "blue")))


galactic_directions = st.builds(
    toward,
    st.floats(min_value=0.0, max_value=360.0, allow_nan=False),
    st.floats(min_value=-90.0, max_value=90.0, allow_nan=False),
)

# Longitudes about the centre of the galaxy, and about the point opposite it.
ABOUT_THE_CENTRE = range(-10, 11, 2)
ABOUT_THE_ANTICENTRE = range(170, 191, 2)
ALL_ROUND = range(0, 360, 5)

# Clear of the dust lane, which lies along the equator, and still inside the thin band.
BESIDE_THE_LANE = 9.0

# Where the two Magellanic Clouds are: galactic longitude and latitude.
MAGELLANIC_CLOUDS = ((280.47, -32.89), (302.80, -44.30))


@given(galactic_directions)
def test_the_light_of_the_galaxy_is_a_colour_a_screen_can_show(direction: GalacticDirection) -> None:
    glow = light(direction)
    assert all(0.0 <= colour <= 1.0 for colour in (glow.red, glow.green, glow.blue))
    assert light(direction) == glow


def test_the_galaxy_is_a_band_along_its_equator_and_dark_at_its_poles() -> None:
    on_the_band = brightness_of(mean_glow(ALL_ROUND, BESIDE_THE_LANE))
    well_off_it = brightness_of(mean_glow(ALL_ROUND, 60.0))
    assert on_the_band > 0.2
    assert well_off_it < on_the_band / 20


def test_the_band_is_brightest_toward_the_centre_of_the_galaxy_and_goes_all_the_way_round() -> None:
    toward_the_centre = brightness_of(mean_glow(ABOUT_THE_CENTRE, BESIDE_THE_LANE))
    opposite = brightness_of(mean_glow(ABOUT_THE_ANTICENTRE, BESIDE_THE_LANE))
    assert toward_the_centre > 3 * opposite
    assert opposite > 0.05


def test_the_bulge_is_gold_and_the_disc_away_from_it_is_blue() -> None:
    bulge = mean_glow(ABOUT_THE_CENTRE, BESIDE_THE_LANE)
    disc = mean_glow(ABOUT_THE_ANTICENTRE, BESIDE_THE_LANE)
    assert bulge.red > bulge.green > bulge.blue
    assert disc.blue > disc.green > disc.red


def test_dust_darkens_the_middle_of_the_band_toward_the_centre_and_takes_the_blue_first() -> None:
    # The middle of the band on the centre's side, six degrees to either side of the equator.
    middle = sorted(
        (light(toward(longitude, half_degrees / 2)) for longitude in range(-40, 41) for half_degrees in range(-12, 13)),
        key=brightness_of,
    )
    darkest, brightest = middle[: len(middle) // 20], middle[-len(middle) // 20 :]
    assert max(map(brightness_of, darkest)) < min(map(brightness_of, brightest)) / 4
    # The star clouds and the grain dim every colour alike. Only dust dims the blue more than the red.
    blueness = lambda glows: math.fsum(glow.blue for glow in glows) / math.fsum(glow.red for glow in glows)
    assert blueness(darkest) < 0.8 * blueness(brightest)


@pytest.mark.parametrize("cloud", MAGELLANIC_CLOUDS)
def test_each_magellanic_cloud_shines_where_the_sky_around_it_is_dark(cloud: tuple[float, float]) -> None:
    longitude, latitude = cloud
    middle = brightness_of(light(toward(longitude, latitude)))
    # The same latitude on the far side of the sky, where there is no cloud.
    elsewhere = brightness_of(light(toward(longitude - 180.0, latitude)))
    assert middle > 0.1
    assert middle > 5 * elsewhere


def test_the_band_is_not_even_but_broken_into_star_clouds() -> None:
    along = [brightness_of(light(toward(longitude, BESIDE_THE_LANE))) for longitude in range(90, 270, 3)]
    assert max(along) > 2 * min(along)


@given(galactic_directions)
def test_noise_over_the_sky_has_no_step_from_one_place_to_the_next(direction: GalacticDirection) -> None:
    # A hundredth of a degree further round the same latitude circle, or over the pole from it.
    nudge = math.radians(0.01)
    turned = GalacticDirection(
        direction.toward_centre * math.cos(nudge) - direction.toward_rotation * math.sin(nudge),
        direction.toward_centre * math.sin(nudge) + direction.toward_rotation * math.cos(nudge),
        direction.toward_pole,
    )
    here = galaxy.smooth_noise(direction, galaxy.CLUMP_FREQUENCY, galaxy.CLUMP_SEED)
    there = galaxy.smooth_noise(turned, galaxy.CLUMP_FREQUENCY, galaxy.CLUMP_SEED)
    assert abs(here - there) < 0.01


def directions_across(longitude: float, latitude: float, pixels: int) -> list[GalacticDirection]:
    """A patch of sky `pixels` wide and high about a place, one direction for each pixel."""
    return [
        toward(longitude + math.degrees(PIXEL) * column, latitude + math.degrees(PIXEL) * row)
        for row in range(pixels)
        for column in range(pixels)
    ]


def lit_share(directions: list[GalacticDirection]) -> float:
    """The share of a patch that a faint star lights, where there is no haze to light it."""
    return sum(brightness_of(light(direction)) > 0.05 for direction in directions) / len(directions)


def test_faint_stars_are_points_scattered_over_the_dark_sky_far_from_the_band() -> None:
    patch = directions_across(40.0, 70.0, 120)
    share = lit_share(patch)
    # Points: nearly all of the patch is dark, and yet there are stars in it.
    assert 0.0005 < share < 0.05
    assert max(brightness_of(light(direction)) for direction in patch) > 0.2


def test_faint_stars_crowd_where_the_band_is_bright() -> None:
    # The edge of the band, above the dust and the thin bright part, against the dark sky far from it.
    def stars_in(patch: list[GalacticDirection]) -> int:
        glows = [brightness_of(light(direction)) for direction in patch]
        floor = sorted(glows)[len(glows) // 2]
        return sum(glow > floor + 0.1 for glow in glows)

    assert stars_in(directions_across(90.0, 12.0, 120)) > 3 * stars_in(directions_across(40.0, 70.0, 120))


def test_a_faint_star_is_no_narrower_than_a_pixel_on_a_coarse_screen() -> None:
    patch = directions_across(40.0, 70.0, 120)
    fine = sum(brightness_of(galaxy.light_toward(direction, Reach(PIXEL / 4))) > 0.05 for direction in patch)
    coarse = sum(brightness_of(galaxy.light_toward(direction, Reach(PIXEL * 2))) > 0.05 for direction in patch)
    assert coarse > fine > 0


def test_a_star_s_dot_is_full_over_its_middle_and_gone_at_its_rim() -> None:
    assert dot_strength(0.0, 0.0) == 1.0
    assert dot_strength(0.5, 0.0) == 1.0
    assert 0.0 < dot_strength(0.9, 0.0) < dot_strength(0.7, 0.0) < 1.0
    assert dot_strength(1.0, 0.0) == 0.0
    # The corners of the quad lie outside the circle, so the dot is round.
    assert dot_strength(1.0, 1.0) == 0.0


def test_the_faintest_catalogue_star_outshines_the_faint_stars_of_the_galaxy() -> None:
    brightest_faint_star = max(sheet.light for sheet in galaxy.STAR_SHEETS)
    faintest = gain_of(Magnitude(catalogue.FAINTEST))
    assert faintest == FAINTEST
    # What reaches the eye at midnight is the gain times the game's fade of a half, in colour and alpha alike.
    assert (faintest / gain_of(Magnitude(catalogue.BRIGHTEST))) ** 2 > 0.6 * brightest_faint_star


def test_the_centre_of_the_galaxy_and_its_pole_lie_where_they_do_among_the_stars() -> None:
    centre = galactic(direction_of(galaxy.CENTRE))
    assert centre.toward_centre == pytest.approx(1.0, abs=1e-4)
    pole = galactic(direction_of(galaxy.NORTH_POLE))
    assert pole.toward_pole == pytest.approx(1.0, abs=1e-9)
    # Deneb lies in the band, a quarter of the way round from the centre: longitude 84.3, latitude 2.0.
    deneb = find(20.690, 45.280)
    seen = galactic(direction_of(SkyPlace(deneb.right_ascension, deneb.declination)))
    expected = toward(84.28, 2.00)
    assert seen.toward_centre == pytest.approx(expected.toward_centre, abs=2e-3)
    assert seen.toward_rotation == pytest.approx(expected.toward_rotation, abs=2e-3)
    assert seen.toward_pole == pytest.approx(expected.toward_pole, abs=2e-3)


def test_the_box_shuts_the_camera_in_with_every_face_toward_it_and_no_corner_beyond_the_stars() -> None:
    corners_met: dict[tuple[int, int, int], int] = {}
    for face in BOX_FACES:
        corners = [box_corner(face, corner) for corner in CORNERS]
        assert faces_the_camera(corners)
        for at in corners:
            assert math.sqrt(dot(at, at)) == pytest.approx(STAR_DISTANCE, abs=1e-9)
            key = (round(math.copysign(1, at.x)), round(math.copysign(1, at.y)), round(math.copysign(1, at.z)))
            corners_met[key] = corners_met.get(key, 0) + 1
    # A quad for each face, and the stars keep the most of the game's quads.
    assert BOX_COUNT == len(BOX_FACES) == len(set(BOX_FACES)) < STAR_COUNT
    # A closed cube: eight corners, and three faces meet at each.
    assert len(corners_met) == 8
    assert set(corners_met.values()) == {3}


def test_the_fragment_stage_paints_the_galaxy_on_the_box_under_the_game_s_fade(game: Game) -> None:
    built = compose(game.shaders, (effect(),))
    vertex, fragment = text_of(built, VERTEX_SHADER), text_of(built, FRAGMENT_SHADER)
    assert f"const int PACK_BOX_COUNT = {BOX_COUNT};" in vertex
    assert "packIsGalaxy = 1;" in vertex
    assert galaxy.GLSL in fragment
    assert "packGalaxyLight(packToward, packPixel), 1.0) * PACK_GALAXY_GAIN * ColorModulator;" in fragment
    assert "fragColor = ColorModulator;" not in fragment


# ---- the effect fits the game ---------------------------------------------------------------------------------------


def test_the_effect_compiles_onto_the_game_s_star_shaders(game: Game) -> None:
    built = compose(game.shaders, (effect(),))
    assert {file.path for file in built} == {VERTEX_SHADER, FRAGMENT_SHADER}
    assert validate(built, game.shaders) == ()


def test_the_effect_is_the_same_each_time_and_changes_only_the_star_shaders() -> None:
    assert effect() == effect()
    assert effect().name == NAME
    assert effect().includes == ()
