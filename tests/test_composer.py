"""The composer joins effects onto the vanilla shaders, and names the effect and the place when it cannot."""

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from zigdag import verifies

from shader.composer import (
    IncludeReplacesVanilla,
    IncludesConflict,
    PlaceMissing,
    PlaceRepeated,
    ReplacementsConflict,
    ShaderUnknown,
    compose,
)
from shader.effects import Attachment, Effect, EffectName, GlslCode, Place, Position
from shader.vanilla import ShaderFile, ShaderPath, ShaderText

FOG = EffectName("fog")
SKY = EffectName("sky")
TERRAIN = ShaderPath("core/terrain.fsh")
SKY_SHADER = ShaderPath("core/sky.fsh")
FOG_INCLUDE = ShaderPath("include/fog_density.glsl")
MAIN = Place("void main() {")
OUTPUT = Place("fragColor = color;")

TERRAIN_TEXT = ShaderText("#version 330\n\nvoid main() {\n    fragColor = color;\n}\n")
TERRAIN_FILE = ShaderFile(TERRAIN, TERRAIN_TEXT)
SKY_FILE = ShaderFile(SKY_SHADER, ShaderText("void main() {\n    fragColor = sky;\n}\n"))


def attach(place: Place, position: Position, code: str, shader: ShaderPath = TERRAIN) -> Attachment:
    return Attachment(shader, place, position, GlslCode(code))


def effect(name: EffectName, *attachments: Attachment, includes: tuple[ShaderFile, ...] = ()) -> Effect:
    return Effect(name, attachments, includes)


def composed_text(*effects: Effect, vanilla: tuple[ShaderFile, ...] = (TERRAIN_FILE,)) -> ShaderText:
    (joined,) = (file for file in compose(vanilla, effects) if file.path == TERRAIN)
    return joined.text


@verifies("shader.composer#join")
def test_before_goes_on_the_lines_before_the_place() -> None:
    fog = effect(FOG, attach(MAIN, Position.BEFORE, "#moj_import <fog.glsl>"))
    assert composed_text(fog) == "#version 330\n\n#moj_import <fog.glsl>\nvoid main() {\n    fragColor = color;\n}\n"


@verifies("shader.composer#join")
def test_after_goes_on_the_lines_after_the_place() -> None:
    fog = effect(FOG, attach(OUTPUT, Position.AFTER, "fragColor.rgb *= 0.5;\nfragColor.a = 1.0;\n"))
    assert composed_text(fog) == (
        "#version 330\n\nvoid main() {\n    fragColor = color;\nfragColor.rgb *= 0.5;\nfragColor.a = 1.0;\n}\n"
    )


@verifies("shader.composer#join")
def test_instead_replaces_the_place() -> None:
    fog = effect(FOG, attach(OUTPUT, Position.INSTEAD, "fragColor = fogged(color);"))
    assert composed_text(fog) == "#version 330\n\nvoid main() {\nfragColor = fogged(color);\n}\n"


@verifies("shader.composer#join")
def test_empty_code_instead_removes_the_place() -> None:
    fog = effect(FOG, attach(OUTPUT, Position.INSTEAD, ""))
    assert composed_text(fog) == "#version 330\n\nvoid main() {\n}\n"


@verifies("shader.composer#join")
def test_code_at_one_place_stacks_in_the_order_of_the_effects() -> None:
    fog = effect(FOG, attach(OUTPUT, Position.BEFORE, "a();"), attach(OUTPUT, Position.AFTER, "c();"))
    sky = effect(
        SKY,
        attach(OUTPUT, Position.BEFORE, "b();"),
        attach(OUTPUT, Position.INSTEAD, "x();"),
        attach(OUTPUT, Position.AFTER, "d();"),
    )
    assert composed_text(fog, sky) == "#version 330\n\nvoid main() {\na();\nb();\nx();\nc();\nd();\n}\n"


@verifies("shader.composer#join")
def test_places_are_found_in_the_original_text() -> None:
    # The sky's code holds the fog's place; the fog still attaches to the game's own line, once.
    sky = effect(SKY, attach(MAIN, Position.AFTER, "fragColor = color;"))
    fog = effect(FOG, attach(OUTPUT, Position.AFTER, "fog();"))
    assert composed_text(sky, fog) == (
        "#version 330\n\nvoid main() {\nfragColor = color;\n    fragColor = color;\nfog();\n}\n"
    )


@verifies("shader.composer#join")
def test_crlf_text_keeps_its_endings_and_code_takes_them() -> None:
    text = ShaderText("void main() {\r\n    fragColor = color;\r\n}\r\n")
    fog = effect(FOG, attach(OUTPUT, Position.AFTER, "a();\nb();"))
    assert composed_text(fog, vanilla=(ShaderFile(TERRAIN, text),)) == (
        "void main() {\r\n    fragColor = color;\r\na();\r\nb();\r\n}\r\n"
    )


@verifies("shader.composer#join")
def test_code_after_a_last_line_without_ending_is_put_on_its_own_line() -> None:
    text = ShaderText("void main() {\n}")
    fog = effect(FOG, attach(Place("}"), Position.AFTER, "// end"))
    assert composed_text(fog, vanilla=(ShaderFile(TERRAIN, text),)) == "void main() {\n}\n// end\n"


@verifies("shader.composer#join")
def test_code_in_a_text_without_endings_ends_with_lf() -> None:
    fog = effect(FOG, attach(Place("}"), Position.BEFORE, "// end"))
    assert composed_text(fog, vanilla=(ShaderFile(TERRAIN, ShaderText("}")),)) == "// end\n}"


@verifies("shader.composer#join")
def test_code_before_a_last_line_without_ending_takes_the_shaders_ending() -> None:
    text = ShaderText("void main() {\r\n}")
    fog = effect(FOG, attach(Place("}"), Position.BEFORE, "// end"))
    assert composed_text(fog, vanilla=(ShaderFile(TERRAIN, text),)) == "void main() {\r\n// end\r\n}"


@verifies("shader.composer#join")
def test_only_changed_shaders_and_includes_are_returned_sorted_by_path() -> None:
    include = ShaderFile(FOG_INCLUDE, ShaderText("float density() { return 1.0; }\n"))
    fog = effect(FOG, attach(MAIN, Position.BEFORE, "#moj_import <fog_density.glsl>"), includes=(include,))
    result = compose((TERRAIN_FILE, SKY_FILE), (fog,))
    assert [file.path for file in result] == [TERRAIN, FOG_INCLUDE]
    assert result[1] == include


@verifies("shader.composer#join")
def test_the_same_include_from_two_effects_is_returned_once() -> None:
    include = ShaderFile(FOG_INCLUDE, ShaderText("float density();\n"))
    result = compose((TERRAIN_FILE,), (effect(FOG, includes=(include,)), effect(SKY, includes=(include,))))
    assert result == (include,)


@verifies("shader.composer#join")
def test_two_includes_of_one_path_with_different_text_name_both_effects() -> None:
    first = ShaderFile(FOG_INCLUDE, ShaderText("float density();\n"))
    second = ShaderFile(FOG_INCLUDE, ShaderText("float density(float height);\n"))
    with pytest.raises(IncludesConflict) as raised:
        compose((TERRAIN_FILE,), (effect(FOG, includes=(first,)), effect(SKY, includes=(second,))))
    assert (raised.value.effect, raised.value.other, raised.value.shader) == (FOG, SKY, FOG_INCLUDE)
    assert "fog" in str(raised.value) and "sky" in str(raised.value) and FOG_INCLUDE in str(raised.value)


@verifies("shader.composer#join")
def test_an_include_cannot_replace_a_vanilla_shader() -> None:
    include = ShaderFile(TERRAIN, ShaderText("void main() {}\n"))
    with pytest.raises(IncludeReplacesVanilla) as raised:
        compose((TERRAIN_FILE,), (effect(FOG, includes=(include,)),))
    assert (raised.value.effect, raised.value.shader) == (FOG, TERRAIN)
    assert "fog" in str(raised.value) and TERRAIN in str(raised.value)


@verifies("shader.composer#join")
def test_an_attachment_to_an_unknown_shader_names_the_effect_and_the_shader() -> None:
    fog = effect(FOG, attach(MAIN, Position.BEFORE, "a();", shader=SKY_SHADER))
    with pytest.raises(ShaderUnknown) as raised:
        compose((TERRAIN_FILE,), (fog,))
    assert (raised.value.effect, raised.value.shader, raised.value.place) == (FOG, SKY_SHADER, MAIN)
    assert "fog" in str(raised.value) and SKY_SHADER in str(raised.value) and MAIN in str(raised.value)


@verifies("shader.composer#join")
def test_two_replacements_of_one_place_name_both_effects() -> None:
    fog = effect(FOG, attach(OUTPUT, Position.INSTEAD, "a();"))
    sky = effect(SKY, attach(OUTPUT, Position.INSTEAD, "b();"))
    with pytest.raises(ReplacementsConflict) as raised:
        compose((TERRAIN_FILE,), (fog, sky))
    error = raised.value
    assert (error.effect, error.other, error.shader, error.place) == (FOG, SKY, TERRAIN, OUTPUT)
    assert all(name in str(error) for name in ("fog", "sky", TERRAIN, OUTPUT))


@verifies("shader.composer#missing-place")
def test_a_missing_place_names_the_effect_and_the_place() -> None:
    gone = Place("fragColor = linear_fog(color);")
    with pytest.raises(PlaceMissing) as raised:
        compose((TERRAIN_FILE,), (effect(FOG, attach(gone, Position.AFTER, "a();")),))
    assert (raised.value.effect, raised.value.shader, raised.value.place) == (FOG, TERRAIN, gone)
    assert all(name in str(raised.value) for name in ("fog", TERRAIN, gone))


@verifies("shader.composer#missing-place")
def test_a_place_must_be_a_whole_line() -> None:
    with pytest.raises(PlaceMissing):
        compose((TERRAIN_FILE,), (effect(FOG, attach(Place("fragColor"), Position.AFTER, "a();")),))


@verifies("shader.composer#repeated-place")
def test_a_repeated_place_names_the_effect_and_the_place() -> None:
    text = ShaderText("void main() {\n    fragColor = color;\n    fragColor = color;\n}\n")
    fog = effect(FOG, attach(OUTPUT, Position.BEFORE, "a();"))
    with pytest.raises(PlaceRepeated) as raised:
        compose((ShaderFile(TERRAIN, text),), (fog,))
    assert (raised.value.effect, raised.value.shader, raised.value.place) == (FOG, TERRAIN, OUTPUT)
    assert all(name in str(raised.value) for name in ("fog", TERRAIN, OUTPUT))


# Lines of GLSL-like text: no newline inside a line, and no space at the ends, so a line is its own place.
line_bodies = st.text(alphabet=st.characters(blacklist_characters="\r\n"), min_size=1).map(str.strip).filter(bool)
endings = st.sampled_from(("\n", "\r\n"))


@st.composite
def shader_texts(draw: st.DrawFn) -> tuple[list[str], ShaderText]:
    bodies = draw(st.lists(line_bodies, min_size=1, unique=True))
    ending = draw(endings)
    return bodies, ShaderText("".join(body + ending for body in bodies))


@verifies("shader.composer#join")
@given(st.lists(st.text(min_size=1), unique=True).map(lambda paths: [ShaderPath(path) for path in paths]))
def test_with_no_effects_nothing_is_returned(paths: list[ShaderPath]) -> None:
    vanilla = tuple(ShaderFile(path, TERRAIN_TEXT) for path in paths)
    assert compose(vanilla, ()) == ()


@verifies("shader.composer#join")
@given(shader_texts(), st.data())
def test_every_line_but_a_replaced_place_survives_in_order(shader: tuple[list[str], ShaderText], data: st.DataObject) -> None:
    bodies, text = shader
    place = data.draw(st.sampled_from(bodies))
    position = data.draw(st.sampled_from(Position))
    code = data.draw(st.text(alphabet=st.characters(blacklist_characters="\r\n")))
    (joined,) = compose((ShaderFile(TERRAIN, text),), (effect(FOG, attach(Place(place), position, code)),))
    kept = [body for body in bodies if not (position is Position.INSTEAD and body == place)]
    remaining = iter(line.removesuffix("\r") for line in joined.text.split("\n"))
    assert all(any(line == body for line in remaining) for body in kept)


@verifies("shader.composer#join")
@given(shader_texts(), st.data())
def test_code_lands_next_to_its_place(shader: tuple[list[str], ShaderText], data: st.DataObject) -> None:
    bodies, text = shader
    index = data.draw(st.integers(min_value=0, max_value=len(bodies) - 1))
    position = data.draw(st.sampled_from(Position))
    code = "// inserted"
    (joined,) = compose((ShaderFile(TERRAIN, text),), (effect(FOG, attach(Place(bodies[index]), position, code)),))
    expected = {
        Position.BEFORE: [*bodies[:index], code, *bodies[index:]],
        Position.AFTER: [*bodies[: index + 1], code, *bodies[index + 1 :]],
        Position.INSTEAD: [*bodies[:index], code, *bodies[index + 1 :]],
    }[position]
    ending = "\r\n" if "\r\n" in text else "\n"
    assert joined.text == "".join(line + ending for line in expected)


@verifies("shader.composer#join")
@given(st.permutations((TERRAIN_FILE, SKY_FILE, ShaderFile(ShaderPath("core/stars.vsh"), TERRAIN_TEXT))))
def test_composing_does_not_depend_on_the_order_of_the_shaders(vanilla: list[ShaderFile]) -> None:
    effects = (
        effect(FOG, attach(OUTPUT, Position.AFTER, "fog();")),
        effect(SKY, attach(MAIN, Position.BEFORE, "sky();", shader=SKY_SHADER)),
    )
    assert compose(tuple(vanilla), effects) == compose((TERRAIN_FILE, SKY_FILE), effects)


@verifies("shader.composer#missing-place")
@given(shader_texts(), line_bodies, st.sampled_from(Position))
def test_any_place_the_shader_lacks_is_named(
    shader: tuple[list[str], ShaderText], place: str, position: Position
) -> None:
    bodies, text = shader
    assume(place not in bodies)
    fog = effect(FOG, attach(Place(place), position, "a();"))
    with pytest.raises(PlaceMissing) as raised:
        compose((ShaderFile(TERRAIN, text),), (fog,))
    assert (raised.value.effect, raised.value.shader, raised.value.place) == (FOG, TERRAIN, place)


@verifies("shader.composer#repeated-place")
@given(shader_texts(), st.data())
def test_any_place_the_shader_repeats_is_named(shader: tuple[list[str], ShaderText], data: st.DataObject) -> None:
    bodies, text = shader
    place = data.draw(st.sampled_from(bodies))
    position = data.draw(st.sampled_from(Position))
    ending = "\r\n" if "\r\n" in text else "\n"
    repeated = ShaderText(text + "    " + place + ending)
    fog = effect(FOG, attach(Place(place), position, "a();"))
    with pytest.raises(PlaceRepeated) as raised:
        compose((ShaderFile(TERRAIN, repeated),), (fog,))
    assert (raised.value.effect, raised.value.shader, raised.value.place) == (FOG, TERRAIN, place)
