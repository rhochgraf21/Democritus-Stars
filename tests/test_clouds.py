"""The clouds are drawn as four forms of real cloud, a form to a place, drifting with the clock and steady far away."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from zigdag import verifies

from shader.composer import compose
from shader.effects.clouds import (
    FRAGMENT_PHASES,
    FRAGMENT_SHADER,
    GLOBALS_INCLUDE,
    NAME,
    VERTEX_COLOUR,
    VERTEX_POSITION,
    VERTEX_SHADER,
    effect,
)
from shader.effects.clouds.forms import (
    FLAT_FOOTPRINT,
    CLOCK_END,
    CLOCK_START,
    DRIFT_SPAN,
    FULL_CLOUD,
    NARROWEST_PIXELS,
    NO_CLOUD,
    NO_GAIN,
    Clock,
    Column,
    Density,
    Footprint,
    Form,
    Gain,
    Grain,
    drift_wrap,
    drifted,
    form_column,
    form_density,
    form_height,
    form_share,
    layer_column,
    layer_density,
    octave_keep,
    place_kind,
    resting_level,
    shape_of,
    tallest_top,
)
from shader.effects.clouds.noise import (
    BILLOW_MEAN,
    ALL,
    HIGHEST,
    MIDDLE,
    NO_WRAP,
    NONE,
    BlockCount,
    BlockSpan,
    Corner,
    CornerIndex,
    Lattice,
    LayerPlace,
    Level,
    Seed,
    Share,
    SizeExponent,
    billow_noise,
    corner_level,
    puff_centre,
    smoothstep,
    value_noise,
)
from shader.effects.clouds.sight import (
    CLEAR,
    DENSEST_ALPHA,
    LIT_HEIGHT,
    LONGEST_TRAVEL,
    MARCH_STEPS,
    NO_STEPS,
    SLAB_HEIGHT,
    SOLID_HEIGHT,
    Alpha,
    Side,
    Travel,
    cloud_alpha,
    lift,
    light_gain,
    looked_into,
    seen_cloud,
    sight_travel,
)
from shader.validator import validate
from shader.vanilla import Game, ShaderFile

# The alpha of the game's cloud colour in the overworld of 26.3 (#ccffffff).
GAME_ALPHA = Alpha(0.8)

# What takes the game's colour of a top, the cloud colour, to its colour of an underside, 0.7 of it.
UNDERSIDE_SHADE = Gain(0.7)

# A line of sight that goes straight up or down moves nowhere over the layer.
STRAIGHT = Travel(BlockSpan(0.0), BlockSpan(0.0))

# How near a line of sight is found to where it goes under a top, as a share of the slab: a look, halved four times.
FOUND_WITHIN = 1.0 / MARCH_STEPS / 16

# How far two results may differ and still be the same number: the sums and products round in the last bits.
CLOSE = 1e-9

# A pixel that covers a twentieth of a block: the camera is close, and every lattice is drawn whole.
NEAR = Footprint(0.05)

# A pixel that covers more of the layer than any lattice is wide: no detail is left to draw.
FAR = Footprint(float(1 << 12))

# One lattice of 64 blocks, for the tests of a single noise.
LATTICE = Lattice(SizeExponent(6), SizeExponent(6))
CORNERS_APART = 1 << LATTICE.east
SEED = Seed(7)

# The world out to where the game lets a player go, and the rest of a place out to where clouds are drawn.
blocks = st.integers(min_value=-30_000_000, max_value=30_000_000)
rests = st.floats(min_value=-2048.0, max_value=2048.0, allow_nan=False)
places = st.builds(LayerPlace, blocks, blocks, rests, rests)
clocks = st.floats(min_value=CLOCK_START, max_value=CLOCK_END, allow_nan=False)
levels = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
forms = st.sampled_from(Form)

# Places a field of 2048 blocks square, 64 blocks apart: enough of a form to hold both cloud and clear sky.
FIELD = tuple(
    LayerPlace(BlockCount(east), BlockCount(south), BlockSpan(0.0), BlockSpan(0.0))
    for east in range(0, 2048, 64)
    for south in range(0, 2048, 64)
)

# Places 2048 blocks apart over a stretch of the world wide enough to hold every form.
WORLD = tuple(
    LayerPlace(BlockCount(east), BlockCount(south), BlockSpan(0.0), BlockSpan(0.0))
    for east in range(-65536, 65536, 2048)
    for south in range(-65536, 65536, 2048)
)


def text_of(files: tuple[ShaderFile, ...], path: str) -> str:
    return next(file.text for file in files if file.path == path)


def at(east: int, south: int) -> LayerPlace:
    return LayerPlace(BlockCount(east), BlockCount(south), BlockSpan(0.0), BlockSpan(0.0))


def stepped(place: LayerPlace, east: float, south: float) -> LayerPlace:
    return LayerPlace(
        place.east_block, place.south_block, BlockSpan(place.east_rest + east), BlockSpan(place.south_rest + south)
    )


# ---- forms: the layer is drawn as cirrus, altocumulus, cumulus and stratus ----------------------------------------


@verifies("shader.effects.clouds#forms")
def test_smooth_noise_at_a_corner_of_its_lattice_is_the_level_of_that_corner() -> None:
    corner = Corner(CornerIndex(3), CornerIndex(-5))
    place = at(3 * CORNERS_APART, -5 * CORNERS_APART)
    assert value_noise(place, LATTICE, NO_WRAP, SEED) == corner_level(corner, SEED)


@verifies("shader.effects.clouds#forms")
@given(places, st.sampled_from((value_noise, billow_noise)))
def test_a_noise_has_no_step_from_one_cell_of_its_lattice_to_the_next(place: LayerPlace, noise) -> None:
    # A step would show as a straight line across the sky. A thousandth of a block is far less than a pixel.
    here = noise(place, LATTICE, NO_WRAP, SEED)
    beside = noise(stepped(place, 1e-3, 1e-3), LATTICE, NO_WRAP, SEED)
    assert abs(here - beside) < 1e-3


@verifies("shader.effects.clouds#forms")
def test_billow_noise_is_highest_at_the_centre_of_a_puff() -> None:
    corner = Corner(CornerIndex(2), CornerIndex(9))
    centre = puff_centre(corner, SEED)
    place = LayerPlace(
        BlockCount(2 * CORNERS_APART),
        BlockCount(9 * CORNERS_APART),
        BlockSpan(centre.east * CORNERS_APART),
        BlockSpan(centre.south * CORNERS_APART),
    )
    assert billow_noise(place, LATTICE, NO_WRAP, SEED) == pytest.approx(HIGHEST, abs=CLOSE)


@verifies("shader.effects.clouds#forms")
@given(places, st.integers(min_value=0, max_value=1000))
def test_a_noise_is_the_same_at_the_same_place_however_the_place_is_split(place: LayerPlace, moved: int) -> None:
    # The camera's block changes as the player walks; a cloud must not.
    other = LayerPlace(
        BlockCount(place.east_block + moved),
        BlockCount(place.south_block - moved),
        BlockSpan(place.east_rest - moved),
        BlockSpan(place.south_rest + moved),
    )
    assert value_noise(other, LATTICE, NO_WRAP, SEED) == pytest.approx(
        value_noise(place, LATTICE, NO_WRAP, SEED), abs=1e-6
    )
    assert billow_noise(other, LATTICE, NO_WRAP, SEED) == pytest.approx(
        billow_noise(place, LATTICE, NO_WRAP, SEED), abs=1e-6
    )


@verifies("shader.effects.clouds#forms")
@pytest.mark.parametrize("form", Form)
def test_every_form_has_cloud_and_clear_sky(form: Form) -> None:
    densities = [form_density(form, place, NEAR) for place in FIELD]
    assert min(densities) == NO_CLOUD
    assert max(densities) > shape_of(form).thickness * MIDDLE


@verifies("shader.effects.clouds#forms")
def test_cirrus_is_the_thinnest_form_and_cumulus_the_thickest() -> None:
    thickness = {form: shape_of(form).thickness for form in Form}
    assert min(thickness, key=thickness.get) is Form.CIRRUS
    assert max(thickness, key=thickness.get) is Form.CUMULUS


@verifies("shader.effects.clouds#forms")
def test_cirrus_runs_in_streaks_along_the_drift() -> None:
    step = BlockSpan(16.0)
    along = sum(
        abs(form_density(Form.CIRRUS, stepped(p, step, 0.0), NEAR) - form_density(Form.CIRRUS, p, NEAR)) for p in FIELD
    )
    across = sum(
        abs(form_density(Form.CIRRUS, stepped(p, 0.0, step), NEAR) - form_density(Form.CIRRUS, p, NEAR)) for p in FIELD
    )
    assert along * 2 < across


@verifies("shader.effects.clouds#forms")
def test_altocumulus_and_cumulus_are_made_of_billows_and_cirrus_and_stratus_are_smooth() -> None:
    assert {form for form in Form if shape_of(form).grain is Grain.BILLOWS} == {Form.ALTOCUMULUS, Form.CUMULUS}
    assert shape_of(Form.ALTOCUMULUS).lattice.east < shape_of(Form.CUMULUS).lattice.east


@verifies("shader.effects.clouds#forms")
def test_a_cloud_hides_nothing_where_it_is_clear_and_what_the_game_s_cloud_hides_where_it_is_full() -> None:
    assert cloud_alpha(GAME_ALPHA, NO_CLOUD) == CLEAR
    assert cloud_alpha(GAME_ALPHA, FULL_CLOUD) == pytest.approx(GAME_ALPHA, abs=CLOSE)
    assert cloud_alpha(Alpha(1.0), Density(0.1)) < DENSEST_ALPHA


@verifies("shader.effects.clouds#forms")
@given(st.floats(min_value=0.05, max_value=1.0), levels, levels)
def test_a_denser_part_of_a_cloud_hides_more(base: float, one: float, other: float) -> None:
    thinner, denser = sorted((one, other))
    assert cloud_alpha(Alpha(base), Density(thinner)) <= cloud_alpha(Alpha(base), Density(denser))


# ---- height: a cloud is not a flat sheet ---------------------------------------------------------------------------


def cumulus_places() -> tuple[LayerPlace, ...]:
    """Places of the world whose clouds are cumulus through and through."""
    return tuple(place for place in WORLD if form_share(place_kind(place), Form.CUMULUS) == ALL)


def around(place: LayerPlace) -> tuple[LayerPlace, ...]:
    """Places 32 blocks apart in the 512 blocks south and east of a place."""
    return tuple(stepped(place, float(east), float(south)) for east in range(0, 512, 32) for south in range(0, 512, 32))


@verifies("shader.effects.clouds#height")
def test_cumulus_stands_the_tallest_and_cirrus_the_lowest() -> None:
    heights = {form: shape_of(form).height for form in Form}
    assert max(heights, key=heights.get) is Form.CUMULUS
    assert min(heights, key=heights.get) is Form.CIRRUS
    assert tallest_top(Level(2 / 3)) == pytest.approx(heights[Form.CUMULUS])


@verifies("shader.effects.clouds#height")
@settings(max_examples=50)
@given(places, forms)
def test_a_top_always_has_cloud_under_it(place: LayerPlace, form: Form) -> None:
    column = form_column(form, place, NEAR)
    assert column.height == form_height(form, place, NEAR)
    assert column.height == NONE or column.density > NO_CLOUD


@verifies("shader.effects.clouds#height")
def test_the_tops_of_cumulus_stand_at_many_heights() -> None:
    heights = {round(form_height(Form.CUMULUS, place, NEAR), 1) for place in FIELD}
    assert len(heights) >= 5


@verifies("shader.effects.clouds#height")
def test_straight_up_from_under_a_cloud_its_base_is_seen_and_from_under_a_gap_the_sky() -> None:
    for home in cumulus_places()[:2]:
        for place in around(home):
            column = layer_column(place, place_kind(place), CLOCK_START, NEAR)
            seen = seen_cloud(place, STRAIGHT, Side.BELOW, MARCH_STEPS, CLOCK_START, NEAR)
            # A thin base is looked through, and the line is under the cloud's own top from the start.
            assert seen.height <= FOUND_WITHIN
            assert seen.density == looked_into(column)


@verifies("shader.effects.clouds#height")
def test_straight_down_onto_a_cloud_its_top_is_seen_where_it_stands() -> None:
    met = 0
    for home in cumulus_places()[:2]:
        for place in around(home):
            column = layer_column(place, place_kind(place), CLOCK_START, NEAR)
            seen = seen_cloud(place, STRAIGHT, Side.ABOVE, MARCH_STEPS, CLOCK_START, NEAR)
            if column.height > NONE:
                met += 1
                assert seen.height == pytest.approx(column.height, abs=FOUND_WITHIN)
    assert met > 0


@verifies("shader.effects.clouds#height")
def test_through_a_gap_the_side_of_a_cloud_behind_is_seen_above_its_base() -> None:
    # Looking east at a slope that crosses the slab in 256 blocks: over some gap a cloud stands in the way.
    slanted = Travel(BlockSpan(256.0), BlockSpan(0.0))
    sides = [
        seen_cloud(place, slanted, Side.BELOW, MARCH_STEPS, CLOCK_START, NEAR)
        for home in cumulus_places()[:2]
        for place in around(home)
        if layer_density(place, CLOCK_START, NEAR) == NO_CLOUD
    ]
    assert any(seen.height > NONE and seen.density == FULL_CLOUD for seen in sides)


@verifies("shader.effects.clouds#height")
def test_with_no_steps_a_cloud_is_drawn_flat_where_the_line_of_sight_enters() -> None:
    slanted = Travel(BlockSpan(256.0), BlockSpan(0.0))
    for place in around(cumulus_places()[0]):
        column = layer_column(place, place_kind(place), CLOCK_START, NEAR)
        below = seen_cloud(place, slanted, Side.BELOW, NO_STEPS, CLOCK_START, NEAR)
        above = seen_cloud(place, slanted, Side.ABOVE, NO_STEPS, CLOCK_START, NEAR)
        assert (below.height, below.density) == (NONE, looked_into(column))
        assert (above.height, above.density) == (column.height, looked_into(column))


@verifies("shader.effects.clouds#height")
def test_a_line_of_sight_moves_over_the_layer_by_its_slope_and_no_further_than_allowed() -> None:
    assert sight_travel(BlockSpan(0.0), BlockSpan(0.0), BlockSpan(100.0)) == STRAIGHT
    level_with_the_drop = sight_travel(BlockSpan(100.0), BlockSpan(0.0), BlockSpan(100.0))
    assert (level_with_the_drop.east, level_with_the_drop.south) == (SLAB_HEIGHT, 0.0)
    nearly_level = sight_travel(BlockSpan(3000.0), BlockSpan(4000.0), BlockSpan(1.0))
    assert (nearly_level.east**2 + nearly_level.south**2) ** 0.5 == pytest.approx(LONGEST_TRAVEL)
    assert nearly_level.east / nearly_level.south == pytest.approx(3000.0 / 4000.0)


@verifies("shader.effects.clouds#height")
def test_a_cloud_with_a_high_top_is_solid_to_look_through_and_a_low_one_as_thin_as_it_is() -> None:
    assert looked_into(Column(Density(0.1), SOLID_HEIGHT)) == FULL_CLOUD
    assert looked_into(Column(Density(0.4), NONE)) == Density(0.4)
    assert Density(0.1) < looked_into(Column(Density(0.1), Share(SOLID_HEIGHT / 2))) < FULL_CLOUD


@verifies("shader.effects.clouds#height")
def test_the_base_of_a_thick_cloud_is_shaded_and_its_side_is_lit_from_a_height_up() -> None:
    assert light_gain(UNDERSIDE_SHADE, lift(NONE, FULL_CLOUD)) == UNDERSIDE_SHADE
    assert light_gain(UNDERSIDE_SHADE, lift(LIT_HEIGHT, FULL_CLOUD)) == NO_GAIN
    assert UNDERSIDE_SHADE < light_gain(UNDERSIDE_SHADE, lift(Share(LIT_HEIGHT / 2), FULL_CLOUD)) < NO_GAIN


@verifies("shader.effects.clouds#forms")
def test_a_thin_part_of_a_cloud_takes_the_colour_of_a_top_at_any_height() -> None:
    assert light_gain(UNDERSIDE_SHADE, lift(NONE, NO_CLOUD)) == NO_GAIN
    assert light_gain(NO_GAIN, lift(NONE, FULL_CLOUD)) == NO_GAIN


@verifies("shader.effects.clouds#flicker")
def test_from_a_footprint_on_the_fragment_stage_draws_the_clouds_flat(game: Game) -> None:
    fragment = text_of(compose(game.shaders, (effect(),)), FRAGMENT_SHADER)
    assert f"const float PACK_FLAT_FOOTPRINT = {FLAT_FOOTPRINT!r};" in fragment
    assert "packDrawnFlat ? PACK_NO_STEPS : PACK_MARCH_STEPS" in fragment


# ---- region: not every place has the same form --------------------------------------------------------------------


@verifies("shader.effects.clouds#region")
def test_every_form_is_the_whole_cloud_of_some_place_of_the_world() -> None:
    kinds = [place_kind(place) for place in WORLD]
    for form in Form:
        assert any(form_share(kind, form) == ALL for kind in kinds), form


@verifies("shader.effects.clouds#region")
@given(levels)
def test_the_forms_of_a_place_add_up_to_one_cloud_of_at_most_two_forms(kind: float) -> None:
    shares = [form_share(Level(kind), form) for form in Form]
    assert sum(shares) == pytest.approx(ALL, abs=CLOSE)
    assert sum(share > NONE for share in shares) <= 2


@verifies("shader.effects.clouds#region")
@given(levels, levels)
def test_a_heavier_kind_of_place_never_has_more_cirrus_or_less_stratus(one: float, other: float) -> None:
    thin, heavy = sorted((one, other))
    assert form_share(Level(heavy), Form.CIRRUS) <= form_share(Level(thin), Form.CIRRUS)
    assert form_share(Level(heavy), Form.STRATUS) >= form_share(Level(thin), Form.STRATUS)


# ---- drift: the clouds move from east to west as the clock advances -----------------------------------------------


@verifies("shader.effects.clouds#drift")
@given(st.integers(min_value=-100_000, max_value=100_000), st.integers(min_value=-100_000, max_value=100_000), forms)
def test_a_quarter_of_the_clock_on_a_cloud_lies_a_quarter_of_the_span_further_west(
    east: int, south: int, form: Form
) -> None:
    quarter = Clock(0.25)
    moved = int(DRIFT_SPAN * quarter)
    before = form_density(form, drifted(at(east, south), CLOCK_START), NEAR)
    after = form_density(form, drifted(at(east - moved, south), quarter), NEAR)
    assert after == pytest.approx(before, abs=CLOSE)


@verifies("shader.effects.clouds#drift")
@given(places, clocks)
def test_the_form_of_a_place_does_not_drift(place: LayerPlace, clock: float) -> None:
    moved = drifted(place, Clock(clock))
    assert (moved.east_block, moved.south_block, moved.south_rest) == (
        place.east_block,
        place.south_block,
        place.south_rest,
    )
    assert moved.east_rest == pytest.approx(place.east_rest + clock * DRIFT_SPAN)


# ---- wrap: nothing changes when the clock starts again --------------------------------------------------------------


@verifies("shader.effects.clouds#wrap")
@settings(max_examples=50)
@given(places)
def test_the_layer_is_the_same_at_the_end_of_the_clock_as_at_its_start(place: LayerPlace) -> None:
    assert layer_density(place, CLOCK_END, NEAR) == pytest.approx(layer_density(place, CLOCK_START, NEAR), abs=1e-6)


@verifies("shader.effects.clouds#wrap")
@pytest.mark.parametrize("form", Form)
def test_every_lattice_of_a_form_repeats_a_whole_number_of_times_in_the_span_of_the_drift(form: Form) -> None:
    shape = shape_of(form)
    for lattice in (shape.field, shape.lattice):
        cells = drift_wrap(lattice) + 1
        assert cells * (1 << lattice.east) == DRIFT_SPAN


# ---- flicker: no detail narrower than 2 pixels ----------------------------------------------------------------------


@verifies("shader.effects.clouds#flicker")
def test_a_lattice_is_left_out_at_two_pixels_and_whole_at_four() -> None:
    apart = BlockSpan(64.0)
    assert octave_keep(apart, Footprint(apart / NARROWEST_PIXELS)) == NONE
    assert octave_keep(apart, Footprint(apart)) == NONE
    assert octave_keep(apart, Footprint(apart / (2 * NARROWEST_PIXELS))) == ALL
    assert NONE < octave_keep(apart, Footprint(apart / 3)) < ALL


@verifies("shader.effects.clouds#flicker")
@given(st.floats(min_value=1.0, max_value=4096.0), st.floats(min_value=0.01, max_value=64.0), st.floats(min_value=1.0, max_value=2.0))
def test_a_lattice_is_never_drawn_more_from_further_away(apart: float, footprint: float, further: float) -> None:
    assert octave_keep(BlockSpan(apart), Footprint(footprint * further)) <= octave_keep(BlockSpan(apart), Footprint(footprint))


@verifies("shader.effects.clouds#flicker")
def test_a_lattice_of_billows_left_out_is_left_at_the_average_of_its_noise() -> None:
    # Left at any other level, the amount of cloud changes where the lattice is left out: a ring around the player.
    # Steps that share no factor with the lattice, so that the places fall evenly over its cells.
    spread = [stepped(at(0, 0), 37.3 * across, 51.7 * down) for across in range(60) for down in range(60)]
    levels = [billow_noise(place, LATTICE, NO_WRAP, SEED) for place in spread]
    assert sum(levels) / len(levels) == pytest.approx(BILLOW_MEAN, abs=0.02)
    assert resting_level(Grain.BILLOWS) == BILLOW_MEAN
    assert resting_level(Grain.SMOOTH) == MIDDLE


@verifies("shader.effects.clouds#flicker")
@settings(max_examples=50)
@given(places, forms)
def test_far_away_a_form_keeps_the_cloud_of_its_field_and_none_of_its_detail(place: LayerPlace, form: Form) -> None:
    shape = shape_of(form)
    field = value_noise(place, shape.field, drift_wrap(shape.field), Seed(shape.seed + 8))
    level = resting_level(shape.grain) + (field - MIDDLE) * shape.field_swing
    expected = smoothstep(shape.clear_below, shape.clear_below + shape.edge, level) * shape.thickness
    assert form_density(form, place, FAR) == pytest.approx(expected, abs=CLOSE)


# ---- the effect fits the game ---------------------------------------------------------------------------------------


@verifies("shader.effects.clouds#forms")
def test_the_effect_compiles_onto_the_game_s_cloud_shaders_in_every_variant(game: Game) -> None:
    built = compose(game.shaders, (effect(),))
    assert {file.path for file in built} == {VERTEX_SHADER, FRAGMENT_SHADER}
    assert validate(built, game.shaders) == ()


@verifies("shader.effects.clouds#forms")
def test_the_vertex_stage_hands_on_where_a_face_lies_before_the_game_projects_it(game: Game) -> None:
    vertex = text_of(compose(game.shaders, (effect(),)), VERTEX_SHADER)
    assert vertex.index(VERTEX_POSITION) < vertex.index("packFromCamera = pos;") < vertex.index("gl_Position = ProjMat")
    assert vertex.index("pos.y += PACK_SLAB_HEIGHT - CellSize.y;") < vertex.index("packFromCamera = pos;")
    assert vertex.index(VERTEX_COLOUR) < vertex.index("gl_Position = PACK_NOWHERE;")


@verifies("shader.effects.clouds#forms")
def test_the_fragment_stage_draws_the_cloud_after_the_distance_fade_and_before_every_phase(game: Game) -> None:
    fragment = text_of(compose(game.shaders, (effect(),)), FRAGMENT_SHADER)
    faded = fragment.index("linear_fog_value(vertexDistance, 0, FogCloudsEnd)")
    drawn = fragment.index("color.a = packCloudAlpha(color.a, packSeen.y);")
    assert faded < drawn < fragment.index(FRAGMENT_PHASES, drawn)
    assert f"const float PACK_DENSEST_ALPHA = {DENSEST_ALPHA!r};" in fragment
    assert f"const float PACK_DRIFT_SPAN = {DRIFT_SPAN!r};" in fragment


@verifies("shader.effects.clouds#drift")
def test_the_fragment_stage_asks_the_game_for_the_clock_and_the_camera_s_block(game: Game) -> None:
    fragment = text_of(compose(game.shaders, (effect(),)), FRAGMENT_SHADER)
    assert fragment.index(GLOBALS_INCLUDE) < fragment.index("void main()")
    assert "CameraBlockPos.xz, packFromCamera.xz - CameraOffset.xz," in fragment
    assert "GameTime, packFootprint);" in fragment


@verifies("shader.effects.clouds#forms")
def test_the_shader_holds_one_shape_for_each_form(game: Game) -> None:
    fragment = text_of(compose(game.shaders, (effect(),)), FRAGMENT_SHADER)
    assert f"const int PACK_FORMS = {len(Form)};" in fragment
    assert fragment.count("    PackShape(ivec2(") == len(Form)


@verifies("shader.effects.clouds#forms")
def test_the_effect_is_the_same_each_time_and_changes_only_the_cloud_shaders() -> None:
    assert effect() == effect()
    assert effect().name == NAME
    assert effect().includes == ()
    assert {attachment.shader for attachment in effect().attachments} == {VERTEX_SHADER, FRAGMENT_SHADER}
