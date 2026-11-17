"""The one command that builds the pack."""

import argparse
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NewType

import deal

from shader import autumn, composer, markers, pack, validator, vanilla
from shader.effects import Effect, EffectName, lighting, sky, stars, weather
from shader.pack import PackFolder
from shader.validator import CompileFailure
from shader.vanilla import Game, JarPath, ResourceFormat, ShaderFile, Texture, TexturePath

# One line of what the command tells the person.
Line = NewType("Line", str)

# What the command hands back to the shell: zero when the pack was written.
ExitCode = NewType("ExitCode", int)

# A moment on the clock the build is timed with, in seconds.
Moment = NewType("Moment", float)

BUILT = ExitCode(0)
NOT_BUILT = ExitCode(1)

# What Python's sys.platform calls a system, such as "darwin".
PlatformName = NewType("PlatformName", str)

MAC = PlatformName("darwin")
WINDOWS = PlatformName("win32")

# The variable Windows keeps the roaming application data folder in, which is where the launcher installs.
ROAMING_VARIABLE = "APPDATA"

# What that variable holds: the path of the roaming folder, as Windows wrote it.
RoamingFolder = NewType("RoamingFolder", str)


@deal.has()
# Off Windows the roaming folder plays no part, whatever the variable happens to hold.
@deal.ensure(lambda _: _.platform == WINDOWS or _.result.is_relative_to(_.home))
def game_folder(platform: PlatformName, home: Path, roaming: RoamingFolder | None) -> Path:
    """Where the launcher keeps the game on a system, which is where a pack is read from and written to.

    `roaming` is the roaming application data folder as Windows names it, or None where none is named.
    """
    if platform == MAC:
        return home / "Library/Application Support/minecraft"
    if platform == WINDOWS:
        return (Path(roaming) if roaming else home / "AppData/Roaming") / ".minecraft"
    # Linux, and any other Unix the game is made to run on, which follow Linux here.
    return home / ".minecraft"


MINECRAFT = game_folder(PlatformName(sys.platform), Path.home(), os.environ.get(ROAMING_VARIABLE))
GAME_JAR = MINECRAFT / "versions/26.3/26.3.jar"
PACK_FOLDER = MINECRAFT / "resourcepacks/zigdag-shaders"

# Order matters where two effects put code at the same place of one shader: it stacks in this order.
EFFECTS: tuple[Callable[[], Effect], ...] = (
    lighting.effect,
    sky.effect,
    stars.effect,
    weather.effect,
)

# What the other parts refuse with. Each becomes a sentence for the person, never a traceback: a compose
# error names the effect and the place, which is how a new game version says what it broke.
REFUSALS = (
    vanilla.VanillaError,
    autumn.MissingColourMap,
    markers.MarkError,
    composer.ComposeError,
    validator.CompilerNotFound,
    validator.IncludeNotFound,
    validator.IncludeCycle,
    pack.NotOurPack,
)


@dataclass(frozen=True)
class Request:
    """What the person asked to be built: from where, to where, and without which effects."""

    jar: JarPath
    pack: PackFolder
    skipped: tuple[EffectName, ...]


@dataclass(frozen=True)
class Build:
    """A pack worked out and not yet compiled or written."""

    resource_format: ResourceFormat
    vanilla: tuple[ShaderFile, ...]
    shaders: tuple[ShaderFile, ...]
    textures: tuple[Texture, ...]


@dataclass(frozen=True)
class Outcome:
    """How a build ended: whether to write the pack, what to tell the person, and the exit code."""

    write: bool
    lines: tuple[Line, ...]
    exit_code: ExitCode


@deal.has()
@deal.post(lambda result: len(result) == len(EFFECTS))
def effect_names() -> tuple[str, ...]:
    """The effects as the person types them to leave one out."""
    return tuple(make().name for make in EFFECTS)


@deal.has()
@deal.ensure(lambda _: _.result.jar == _.jar and _.result.pack == _.folder and len(_.result.skipped) == len(_.skipped))
def requested(jar: Path, folder: Path, skipped: list[str]) -> Request:
    """The person's words as a request."""
    return Request(JarPath(jar), PackFolder(folder), tuple(EffectName(name) for name in skipped))


@deal.has()
@deal.post(lambda result: set(autumn.COLOUR_MAPS) <= set(result) and set(markers.MARKED) <= set(result))
def wanted() -> tuple[TexturePath, ...]:
    """The textures a build asks the game for."""
    return (*autumn.COLOUR_MAPS, *markers.MARKED)


@deal.has()
# The game as vanilla reads it: every texture has the pixels its size says, which the autumn colours count on.
@deal.pre(lambda request, game: all(autumn.is_well_formed(texture) for texture in game.textures))
@deal.raises(autumn.MissingColourMap, markers.MarkError, composer.ComposeError)
@deal.ensure(lambda _: _.result.vanilla == _.game.shaders and _.result.resource_format == _.game.resource_format)
def planned(request: Request, game: Game) -> Build:
    """The pack for the request: the autumn colours, the marked textures and the effects on the game's shaders."""
    colours = autumn.colour_maps(game.textures)
    marked = markers.mark(tuple(texture for texture in game.textures if texture.path in markers.MARKED))
    effects = tuple(made for made in (make() for make in EFFECTS) if made.name not in request.skipped)
    return Build(game.resource_format, game.shaders, composer.compose(game.shaders, effects), colours + marked)


@deal.has()
@deal.ensure(lambda _: _.result.write == (not _.failures))
@deal.post(lambda result: result.write == (result.exit_code == BUILT))
@deal.post(lambda result: len(result.lines) > 0)
def judged(
    request: Request, game: Game, build: Build, failures: tuple[CompileFailure, ...], started: Moment, ended: Moment
) -> Outcome:
    """Whether the build may be written, and what to say about it."""
    if failures:
        # One shader that does not compile makes the game switch the whole pack off, so nothing is written.
        told = tuple(Line(f"{failure.shader} {failure.defines}:\n{failure.message}") for failure in failures)
        return Outcome(False, (*told, Line(f"not written: {len(failures)} shaders do not compile")), NOT_BUILT)
    summary = Line(
        f"{request.pack}: game {game.version}, {len(build.shaders)} shaders, {ended - started:.1f} s"
    )
    return Outcome(True, (summary,), BUILT)


@deal.has()
@deal.post(lambda result: not result.write and result.exit_code == NOT_BUILT and len(result.lines) == 1)
def refused(refusal: Exception) -> Outcome:
    """What to say when a part refuses: its own words, and nothing written."""
    return Outcome(False, (Line(f"not written: {type(refusal).__name__}: {refusal}"),), NOT_BUILT)


def build(request: Request) -> ExitCode:
    """Read the game, work the pack out, compile it, write it if every shader compiles, and say how it went."""
    started = Moment(time.perf_counter())
    try:
        game = vanilla.read_game(request.jar, wanted())
        plan = planned(request, game)
        failures = validator.validate(plan.shaders, plan.vanilla)
        outcome = judged(request, game, plan, failures, started, Moment(time.perf_counter()))
        if outcome.write:
            pack.write_pack(request.pack, plan.resource_format, plan.shaders, plan.textures)
    except REFUSALS as refusal:
        outcome = refused(refusal)
    for line in outcome.lines:
        print(line)
    return outcome.exit_code


def main() -> int:
    """Build the pack the command line asks for."""
    parser = argparse.ArgumentParser(prog="shader", description=__doc__)
    parser.add_argument("--jar", type=Path, default=GAME_JAR)
    parser.add_argument("--pack", type=Path, default=PACK_FOLDER)
    parser.add_argument("--skip", action="append", default=[], choices=effect_names(), help="leave an effect out")
    arguments = parser.parse_args()
    return build(requested(arguments.jar, arguments.pack, arguments.skip))


if __name__ == "__main__":
    sys.exit(main())
