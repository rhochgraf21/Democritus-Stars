"""The validator compiles each built shader under each combination of the defines it tests, and names what fails."""

import os
import time
import zipfile

import pytest
from zigdag import verifies

from shader.validator import CompilerNotFound, IncludeCycle, IncludeNotFound, validate
from shader.vanilla import ShaderFile, ShaderPath, ShaderText

# Where a game jar keeps its shader files.
JAR_SHADERS = "assets/minecraft/shaders/"
GAME_JAR = "SHADER_GAME_JAR"  # the environment variable that names a game jar to check the vanilla shaders of


def shader(path: str, *lines: str) -> ShaderFile:
    return ShaderFile(ShaderPath(path), ShaderText("\n".join(lines) + "\n"))


GOOD = shader(
    "core/good.fsh",
    "#version 330",
    "layout(location = 0) out vec4 fragColor;",
    "void main() {",
    "    fragColor = vec4(1.0);",
    "}",
)

LIBRARY = shader("include/tint.glsl", "vec4 tint() {", "    return vec4(0.5);", "}")
BROKEN_LIBRARY = shader("include/tint.glsl", "vec4 tint() {", "    return vec4(0.5) oops;", "}")


def tinted(path: str) -> ShaderFile:
    return shader(
        path,
        "#version 330",
        "#include <minecraft:tint.glsl>",
        "layout(location = 0) out vec4 fragColor;",
        "void main() {",
        "    fragColor = tint();",
        "}",
    )


@verifies("shader.validator#compile")
def test_a_good_shader_compiles() -> None:
    assert validate((GOOD,), ()) == ()


@verifies("shader.validator#compile")
def test_an_error_under_one_define_is_named_with_that_define() -> None:
    broken = shader(
        "core/broken.fsh",
        "#version 330",
        "layout(location = 0) out vec4 fragColor;",
        "void main() {",
        "    fragColor = vec4(1.0);",
        "#ifdef GLINT",
        "    fragColor = vec4(1.0) oops;",
        "#endif",
        "#ifdef EMISSIVE",
        "    fragColor *= 2.0;",
        "#endif",
        "}",
    )
    (failure,) = validate((broken,), ())
    assert failure.shader == "core/broken.fsh"
    assert failure.defines == ("GLINT",)
    assert "core/broken.fsh:6:" in failure.message


@verifies("shader.validator#compile")
def test_an_error_under_two_defines_together_is_found() -> None:
    broken = shader(
        "core/both.fsh",
        "#version 330",
        "layout(location = 0) out vec4 fragColor;",
        "void main() {",
        "    fragColor = vec4(1.0);",
        "#if defined(OIT) && !defined(OIT_ALPHA_ONLY)",
        "  #ifdef GLINT",
        "    fragColor = undeclared;",
        "  #endif",
        "#endif",
        "}",
    )
    (failure,) = validate((broken,), ())
    assert failure.defines == ("GLINT", "OIT")
    assert "core/both.fsh:7:" in failure.message


@verifies("shader.validator#compile")
def test_a_broken_include_fails_every_shader_that_includes_it() -> None:
    vanilla = (LIBRARY, tinted("core/first.fsh"), tinted("core/second.fsh"), GOOD)
    failures = validate((BROKEN_LIBRARY,), vanilla)
    assert {f.shader for f in failures} == {"core/first.fsh", "core/second.fsh"}
    assert all("include/tint.glsl:2:" in f.message for f in failures)


@verifies("shader.validator#compile")
def test_a_built_include_wins_over_the_vanilla_one() -> None:
    assert validate((LIBRARY, tinted("core/first.fsh")), (BROKEN_LIBRARY,)) == ()


@verifies("shader.validator#compile")
def test_a_cutout_compiles_as_the_number_the_game_gives() -> None:
    cutout = shader(
        "core/cutout.fsh",
        "#version 330",
        "layout(location = 0) out vec4 fragColor;",
        "void main() {",
        "    fragColor = vec4(1.0);",
        "#ifdef ALPHA_CUTOUT",
        "    if (fragColor.a < ALPHA_CUTOUT) {",
        "        discard;",
        "    }",
        "#endif",
        "}",
    )
    assert validate((cutout,), ()) == ()


def with_main(path: str, colour: str) -> ShaderFile:
    return shader(
        path,
        "#version 330",
        "layout(location = 0) out vec4 fragColor;",
        "void main() {",
        f"    fragColor = {colour};",
        "#ifdef UNUSED_BY_THE_GAME",
        "    fragColor = missing();",
        "#endif",
        "}",
    )


@verifies("shader.validator#compile")
def test_a_combination_that_fails_for_vanilla_too_is_not_a_failure_of_the_pack() -> None:
    vanilla = with_main("core/old.fsh", "vec4(1.0)")
    built = with_main("core/old.fsh", "vec4(0.5)")
    assert validate((built,), (vanilla,)) == ()


@verifies("shader.validator#compile")
def test_a_new_shader_is_not_excused_by_a_vanilla_original_it_does_not_have() -> None:
    (failure,) = validate((with_main("core/new.fsh", "vec4(0.5)"),), ())
    assert failure.defines == ("UNUSED_BY_THE_GAME",)


def never(path: str, *included: str) -> ShaderFile:
    return shader(path, "#version 330", *included, "void main() {", "    undeclared = 1;", "}")


@verifies("shader.validator#compile")
def test_nothing_is_excused_when_the_original_compiles_under_no_combination() -> None:
    (failure,) = validate((never("core/never.fsh"),), (never("core/never.fsh"),))
    assert "core/never.fsh:3:" in failure.message


@verifies("shader.validator#compile")
def test_a_vanilla_shader_that_reaches_no_changed_include_is_not_compiled() -> None:
    # A vanilla shader that compiles under no combination would be reported, were it compiled.
    unchanged = (LIBRARY, never("core/tinted.fsh", "#include <minecraft:tint.glsl>"))
    assert validate((LIBRARY,), unchanged) == ()
    other = shader("include/other.glsl", "float other() {", "    return 1.0;", "}")
    elsewhere = (LIBRARY, other, tinted("core/first.fsh"), never("core/never.fsh"))
    assert validate((shader("include/other.glsl", "float other() {", "    return 0.5;", "}"),), elsewhere) == ()


def switched(path: str, glint: str, emissive: str) -> ShaderFile:
    return shader(
        path,
        "#version 330",
        "layout(location = 0) out vec4 fragColor;",
        "void main() {",
        "    fragColor = vec4(1.0);",
        "#ifdef EMISSIVE",
        f"    {emissive}",
        "#endif",
        "#ifdef GLINT",
        f"    {glint}",
        "#endif",
        "}",
    )


@verifies("shader.validator#compile")
def test_a_combination_the_original_compiles_under_is_held_against_the_pack() -> None:
    vanilla = switched("core/glint.fsh", "fragColor *= 0.5;", "fragColor *= 2.0;")
    built = switched("core/glint.fsh", "fragColor = missing();", "fragColor *= 2.0;")
    (failure,) = validate((built,), (vanilla,))
    assert failure.defines == ("GLINT",)


@verifies("shader.validator#compile")
def test_an_excused_combination_does_not_hide_another_failure() -> None:
    vanilla = switched("core/glint.fsh", "fragColor *= 0.5;", "fragColor = old();")
    built = switched("core/glint.fsh", "fragColor = missing();", "fragColor = old();")
    (failure,) = validate((built,), (vanilla,))
    assert failure.defines == ("GLINT",)


@verifies("shader.validator#compile")
def test_a_missing_include_is_named() -> None:
    with pytest.raises(IncludeNotFound) as raised:
        validate((tinted("core/first.fsh"),), ())
    assert (raised.value.include, raised.value.by) == ("include/tint.glsl", "core/first.fsh")


@verifies("shader.validator#compile")
def test_an_include_that_includes_itself_is_named() -> None:
    loop = shader("include/tint.glsl", "#include <minecraft:tint.glsl>")
    with pytest.raises(IncludeCycle) as raised:
        validate((loop, tinted("core/first.fsh")), ())
    assert (raised.value.include, raised.value.by) == ("include/tint.glsl", "include/tint.glsl")


@verifies("shader.validator#compile")
def test_a_missing_compiler_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    with pytest.raises(CompilerNotFound):
        validate((GOOD,), ())


def game_shaders(jar: str) -> tuple[ShaderFile, ...]:
    with zipfile.ZipFile(jar) as archive:
        return tuple(
            ShaderFile(ShaderPath(name.removeprefix(JAR_SHADERS)), ShaderText(archive.read(name).decode()))
            for name in archive.namelist()
            if name.startswith(JAR_SHADERS) and not name.endswith("/")
        )


# Not linked to the statement: zigdag takes a skipped test for one that ran no code of the part, and this one is
# skipped wherever the game jar is not at hand.
@pytest.mark.skipif(GAME_JAR not in os.environ, reason=f"{GAME_JAR} names no game jar")
def test_every_untouched_vanilla_core_shader_compiles() -> None:
    # Built and vanilla are the same files, so a variant that fails is excused only where the game does not use it;
    # a shader that compiled under no combination at all would still be reported.
    vanilla = game_shaders(os.environ[GAME_JAR])
    core = tuple(f for f in vanilla if f.path.startswith("core/"))
    started = time.monotonic()
    assert validate(core, vanilla) == ()
    print(f"validated {len(core)} vanilla core shaders in {time.monotonic() - started:.1f} s")
