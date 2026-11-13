"""The clouds draw each cell as a puff that fades to nothing at its edge, and touching puffs leave no gap."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from zigdag import verifies

from shader.composer import compose
from shader.effects import Position
from shader.effects.clouds import (
    CELL_CENTRE,
    CENTRE,
    CLEAR,
    DENSEST_ALPHA,
    EDGE,
    FRAGMENT_PHASES,
    FRAGMENT_SHADER,
    FULL_CLOUD,
    NAME,
    NO_CLOUD,
    OPAQUE,
    PUFF_HALF_SIZE,
    SHARED_DEPTH_BITS,
    VERTEX_COLOUR,
    VERTEX_POSITION,
    VERTEX_SHADER,
    Alpha,
    BitCount,
    CellSpan,
    Density,
    LayerPoint,
    PuffPosition,
    Reach,
    cleared_depth_bits,
    cloud_alpha,
    cloud_density,
    effect,
    falloff,
    puff_alpha,
    puff_density,
    puff_position,
)
from shader.validator import validate
from shader.vanilla import Game, ShaderFile

# The alpha of the game's cloud colour in the overworld of 26.3 (#ccffffff).
GAME_ALPHA = Alpha(0.8)

# How far two results may differ and still be the same number: the sums and powers round in the last bits.
CLOSE = 1e-12

# The distance between the centres of two cells that share an edge.
ONE_CELL = CellSpan(1.0)

# Well inside a puff: nearer the edge than this, a density is too small to tell from none.
WELL_INSIDE = Reach(0.999)

# The smallest step between two reaches that the falloff must tell apart.
STEP = 1e-3

HERE = LayerPoint(CellSpan(0.0), CellSpan(0.0))
EAST = LayerPoint(ONE_CELL, CellSpan(0.0))
SOUTH = LayerPoint(CellSpan(0.0), ONE_CELL)
SOUTH_EAST = LayerPoint(ONE_CELL, ONE_CELL)
BLOCK = (HERE, EAST, SOUTH, SOUTH_EAST)

reaches = st.floats(min_value=-2.0, max_value=2.0, allow_nan=False)
inside = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
between = st.floats(min_value=0.0, max_value=ONE_CELL, allow_nan=False)
alphas = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
visible_alphas = st.floats(min_value=0.05, max_value=1.0, allow_nan=False)


def at(east: float, south: float) -> LayerPoint:
    return LayerPoint(CellSpan(east), CellSpan(south))


# ---- puff: each cell is a puff that fades to nothing at its edge --------------------------------------------------


@verifies("shader.effects.clouds#puff")
@pytest.mark.parametrize(
    ("reach", "density"),
    [(0.0, 1.0), (0.25, 0.84375), (0.5, 0.5), (0.75, 0.15625), (1.0, 0.0), (-0.25, 0.84375), (-1.0, 0.0)],
)
def test_the_falloff_is_the_smoothstep_curve_turned_over(reach: float, density: float) -> None:
    assert falloff(Reach(reach)) == density


@verifies("shader.effects.clouds#puff")
@given(reaches)
def test_a_puff_is_densest_at_its_centre(reach: float) -> None:
    assert falloff(Reach(reach)) <= falloff(CENTRE) == FULL_CLOUD


@verifies("shader.effects.clouds#puff")
@given(inside, inside)
def test_a_puff_never_gets_denser_toward_its_edge(one: float, other: float) -> None:
    near, far = sorted((one, other))
    assert falloff(Reach(near)) >= falloff(Reach(far))
    if far - near >= STEP:
        assert falloff(Reach(near)) > falloff(Reach(far))


@verifies("shader.effects.clouds#puff")
@given(st.floats(min_value=EDGE, max_value=1e6, allow_nan=False), st.sampled_from((1.0, -1.0)))
def test_a_puff_is_nothing_at_its_edge_and_beyond(distance: float, side: float) -> None:
    assert falloff(Reach(side * distance)) == NO_CLOUD


@verifies("shader.effects.clouds#puff")
@given(st.floats(min_value=-WELL_INSIDE, max_value=WELL_INSIDE, allow_nan=False))
def test_a_puff_is_something_everywhere_inside_its_edge(reach: float) -> None:
    assert falloff(Reach(reach)) > NO_CLOUD
    assert falloff(Reach(reach)) == falloff(Reach(-reach))


@verifies("shader.effects.clouds#puff")
def test_a_puff_reaches_from_its_centre_to_the_centres_of_the_cells_beside_it() -> None:
    assert PUFF_HALF_SIZE == ONE_CELL
    assert puff_position(at(0.5, -0.25), HERE) == PuffPosition(Reach(0.5), Reach(-0.25))
    assert puff_position(at(1.25, 1.5), SOUTH_EAST) == PuffPosition(Reach(0.25), Reach(0.5))
    assert puff_position(EAST, HERE) == PuffPosition(EDGE, CENTRE)
    assert puff_position(HERE, SOUTH) == PuffPosition(CENTRE, Reach(-EDGE))


@verifies("shader.effects.clouds#puff")
@pytest.mark.parametrize(
    ("across", "along", "density"),
    [(0.0, 0.0, 1.0), (0.5, 0.0, 0.5), (0.0, -0.5, 0.5), (0.5, 0.5, 0.25), (0.25, -0.5, 0.421875)],
)
def test_the_density_of_a_puff_is_its_falloff_along_both_axes(across: float, along: float, density: float) -> None:
    assert puff_density(PuffPosition(Reach(across), Reach(along))) == density


@verifies("shader.effects.clouds#puff")
@given(reaches, reaches)
def test_a_puff_is_nothing_on_every_side_of_its_edge(across: float, along: float) -> None:
    density = puff_density(PuffPosition(Reach(across), Reach(along)))
    assert density <= puff_density(PuffPosition(CENTRE, CENTRE)) == FULL_CLOUD
    if max(abs(across), abs(along)) >= EDGE:
        assert density == NO_CLOUD
    if max(abs(across), abs(along)) <= WELL_INSIDE:
        assert density > NO_CLOUD


@verifies("shader.effects.clouds#puff")
def test_a_puff_hides_what_the_game_s_cloud_hides_at_its_centre_and_nothing_at_its_edge() -> None:
    assert puff_alpha(GAME_ALPHA, FULL_CLOUD) == pytest.approx(GAME_ALPHA, abs=CLOSE)
    assert puff_alpha(GAME_ALPHA, NO_CLOUD) == CLEAR
    assert puff_alpha(GAME_ALPHA, Density(0.5)) == pytest.approx(1 - math.sqrt(1 - GAME_ALPHA), abs=CLOSE)
    assert puff_alpha(CLEAR, FULL_CLOUD) == CLEAR


@verifies("shader.effects.clouds#puff")
def test_an_opaque_cloud_still_fades_at_the_edge_of_a_puff() -> None:
    assert DENSEST_ALPHA == 0.95
    assert puff_alpha(OPAQUE, FULL_CLOUD) == pytest.approx(DENSEST_ALPHA, abs=CLOSE)
    assert puff_alpha(OPAQUE, Density(0.5)) == pytest.approx(1 - math.sqrt(1 - DENSEST_ALPHA), abs=CLOSE)
    assert puff_alpha(OPAQUE, Density(0.01)) < 0.05


@verifies("shader.effects.clouds#puff")
@given(visible_alphas, inside, inside)
def test_a_denser_part_of_a_puff_hides_more(base: float, one: float, other: float) -> None:
    thin, thick = sorted((one, other))
    assert puff_alpha(Alpha(base), Density(thin)) <= puff_alpha(Alpha(base), Density(thick))
    if thick - thin >= STEP:
        assert puff_alpha(Alpha(base), Density(thin)) < puff_alpha(Alpha(base), Density(thick))


@verifies("shader.effects.clouds#puff")
@given(alphas, st.floats(min_value=-3.0, max_value=3.0, allow_nan=False), st.sampled_from((1.0, -1.0)))
def test_a_cloud_of_one_cell_fades_to_nothing_at_the_edge_of_its_puff(base: float, along: float, side: float) -> None:
    on_the_edge = at(side * PUFF_HALF_SIZE, along)
    assert cloud_alpha(Alpha(base), on_the_edge, (HERE,)) == CLEAR
    assert cloud_alpha(Alpha(base), at(along, side * PUFF_HALF_SIZE), (HERE,)) == CLEAR
    assert cloud_alpha(Alpha(base), HERE, (HERE,)) == pytest.approx(min(base, DENSEST_ALPHA), abs=CLOSE)


@verifies("shader.effects.clouds#puff")
def test_no_cells_are_no_cloud() -> None:
    assert cloud_density(HERE, ()) == NO_CLOUD
    assert cloud_alpha(GAME_ALPHA, HERE, ()) == CLEAR


# ---- overlap: no gap between the puffs of two cells that touch -----------------------------------------------------
#
# Two cells touch when they share an edge: those are the neighbours the game itself knows, and between
# which it leaves out the side faces. "No gap" is held to the strictest reading there is: everywhere
# between the two centres the cloud is as dense, and hides as much, as at the centre of a single puff.


@verifies("shader.effects.clouds#overlap")
@given(between)
def test_between_the_centres_of_two_touching_cells_the_cloud_is_as_dense_as_at_a_centre(along: float) -> None:
    alone = cloud_density(HERE, (HERE,))
    assert cloud_density(at(along, 0.0), (HERE, EAST)) == pytest.approx(alone, abs=CLOSE)
    assert cloud_density(at(0.0, along), (HERE, SOUTH)) == pytest.approx(alone, abs=CLOSE)


@verifies("shader.effects.clouds#overlap")
@given(visible_alphas, between)
def test_between_the_centres_of_two_touching_cells_the_cloud_hides_as_much_as_at_a_centre(
    base: float, along: float
) -> None:
    alone = cloud_alpha(Alpha(base), HERE, (HERE,))
    assert cloud_alpha(Alpha(base), at(along, 0.0), (HERE, EAST)) == pytest.approx(alone, abs=CLOSE)
    assert cloud_alpha(Alpha(base), at(0.0, along), (SOUTH, HERE)) == pytest.approx(alone, abs=CLOSE)


@verifies("shader.effects.clouds#overlap")
def test_a_single_puff_would_have_thinned_to_half_where_two_touching_cells_meet() -> None:
    midpoint = at(ONE_CELL * CELL_CENTRE, 0.0)
    assert cloud_density(midpoint, (HERE,)) == 0.5
    assert cloud_density(midpoint, (HERE, EAST)) == FULL_CLOUD


@verifies("shader.effects.clouds#overlap")
@given(between, st.floats(min_value=-2.0, max_value=2.0, allow_nan=False))
def test_two_touching_cells_fade_sideways_as_one_puff_does(along: float, sideways: float) -> None:
    pair = cloud_density(at(along, sideways), (HERE, EAST))
    assert pair == pytest.approx(falloff(Reach(sideways / PUFF_HALF_SIZE)), abs=CLOSE)


@verifies("shader.effects.clouds#overlap")
@given(visible_alphas, between, between)
def test_a_cloud_is_even_wherever_the_four_cells_around_a_point_are_cloud(
    base: float, east: float, south: float
) -> None:
    point = at(east, south)
    assert cloud_density(point, BLOCK) == pytest.approx(FULL_CLOUD, abs=CLOSE)
    assert cloud_alpha(Alpha(base), point, BLOCK) == pytest.approx(min(base, DENSEST_ALPHA), abs=CLOSE)


@verifies("shader.effects.clouds#overlap")
def test_the_order_the_puffs_are_drawn_in_does_not_change_the_cloud() -> None:
    point = at(0.3, 0.6)
    forward = cloud_alpha(GAME_ALPHA, point, (HERE, EAST, SOUTH))
    backward = cloud_alpha(GAME_ALPHA, point, (SOUTH, EAST, HERE))
    assert forward == pytest.approx(backward, abs=CLOSE)
    assert forward == pytest.approx(1 - (1 - GAME_ALPHA) ** cloud_density(point, (HERE, EAST, SOUTH)), abs=CLOSE)


@verifies("shader.effects.clouds#overlap")
def test_cells_that_only_meet_at_a_corner_are_joined_by_a_thinner_bridge() -> None:
    corner = at(ONE_CELL * CELL_CENTRE, ONE_CELL * CELL_CENTRE)
    assert cloud_density(corner, (HERE, SOUTH_EAST)) == 0.5
    assert cloud_alpha(GAME_ALPHA, corner, (HERE, SOUTH_EAST)) == pytest.approx(
        1 - math.sqrt(1 - GAME_ALPHA), abs=CLOSE
    )


# ---- the depth that the puffs of one plane share -------------------------------------------------------------------


@verifies("shader.effects.clouds#overlap")
def test_the_bits_of_a_depth_that_puffs_may_differ_in_are_its_lowest() -> None:
    assert cleared_depth_bits(BitCount(0)) == 0
    assert cleared_depth_bits(BitCount(1)) == 0b1
    assert cleared_depth_bits(BitCount(3)) == 0b111
    assert cleared_depth_bits(SHARED_DEPTH_BITS) == 0xFFFF


# ---- the effect fits the game ---------------------------------------------------------------------------------------


def text_of(files: tuple[ShaderFile, ...], path: str) -> str:
    return next(file.text for file in files if file.path == path)


@verifies("shader.effects.clouds#puff")
def test_the_effect_compiles_onto_the_game_s_cloud_shaders_in_every_variant(game: Game) -> None:
    built = compose(game.shaders, (effect(),))
    assert {file.path for file in built} == {VERTEX_SHADER, FRAGMENT_SHADER}
    assert validate(built, game.shaders) == ()


@verifies("shader.effects.clouds#puff")
def test_the_vertex_stage_grows_the_face_before_the_game_projects_it(game: Game) -> None:
    vertex = text_of(compose(game.shaders, (effect(),)), VERTEX_SHADER)
    assert vertex.index(VERTEX_POSITION) < vertex.index("pos.xz = ") < vertex.index("gl_Position = ProjMat")
    assert vertex.index(VERTEX_COLOUR) < vertex.index("gl_Position = PACK_NOWHERE;")
    assert f"const float PACK_PUFF_HALF_SIZE = {PUFF_HALF_SIZE!r};" in vertex
    assert f"const float PACK_CELL_CENTRE = {CELL_CENTRE!r};" in vertex


@verifies("shader.effects.clouds#puff")
def test_the_fragment_stage_shapes_the_alpha_after_the_distance_fade_and_before_every_phase(game: Game) -> None:
    fragment = text_of(compose(game.shaders, (effect(),)), FRAGMENT_SHADER)
    faded = fragment.index("linear_fog_value(vertexDistance, 0, FogCloudsEnd)")
    shaped = fragment.index("color.a = packPuffAlpha(color.a, packPuffDensity(packPuffPosition));")
    assert faded < shaped < fragment.index(FRAGMENT_PHASES, shaped)
    assert f"const float PACK_DENSEST_ALPHA = {DENSEST_ALPHA!r};" in fragment
    assert f"const float PACK_EDGE = {EDGE!r};" in fragment
    assert "const int PACK_CLEARED_DEPTH_BITS = 0xffff;" in fragment


@verifies("shader.effects.clouds#puff")
def test_the_effect_is_the_same_each_time_and_changes_only_the_cloud_shaders() -> None:
    assert effect() == effect()
    assert effect().name == NAME
    assert effect().includes == ()
    assert [attachment.shader for attachment in effect().attachments] == [VERTEX_SHADER] * 3 + [FRAGMENT_SHADER] * 2
    assert [attachment.position for attachment in effect().attachments] == [Position.AFTER] * 4 + [Position.BEFORE]
