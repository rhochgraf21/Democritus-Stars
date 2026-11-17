"""The one command that builds the pack for a season."""

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NewType

import deal

from shader import cloudmap, composer, markers, pack, seasons, validator, vanilla
from shader.effects import Effect, EffectName, clouds, lighting, sky, stars, weather
from shader.pack import PackFolder
from shader.seasons import Season
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

MINECRAFT = Path.home() / "Library/Application Support/minecraft"
GAME_JAR = MINECRAFT / "versions/26.3/26.3.jar"
PACK_FOLDER = MINECRAFT / "resourcepacks/zigdag-shaders"

# Order matters where two effects put code at the same place of one shader: it stacks in this order.
EFFECTS: tuple[Callable[[], Effect], ...] = (
    lighting.effect,
    sky.effect,
    stars.effect,
    clouds.effect,
    weather.effect,
)

# What the other parts refuse with. Each becomes a sentence for the person, never a traceback: a compose
# error names the effect and the place, which is how a new game version says what it broke.
REFUSALS = (
    vanilla.VanillaError,
    seasons.MissingColourMap,
    cloudmap.MissingCloudMap,
    markers.MarkError,
    composer.ComposeError,
    validator.CompilerNotFound,
    validator.IncludeNotFound,
    validator.IncludeCycle,
    pack.NotOurPack,
)


@dataclass(frozen=True)
class Request:
    """What the person asked to be built, from where, to where, and without which effects."""

    season: Season
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
@deal.post(lambda result: len(result) == len(Season))
def season_names() -> tuple[str, ...]:
    """The seasons as the person types them."""
    return tuple(season.name.lower() for season in Season)


@deal.has()
@deal.post(lambda result: len(result) == len(EFFECTS))
def effect_names() -> tuple[str, ...]:
    """The effects as the person types them to leave one out."""
    return tuple(make().name for make in EFFECTS)


@deal.has()
@deal.pre(lambda season, jar, folder, skipped: season in season_names())
@deal.ensure(lambda _: _.result.jar == _.jar and _.result.pack == _.folder and len(_.result.skipped) == len(_.skipped))
@deal.ensure(lambda _: _.result.season.name.lower() == _.season)
def requested(season: str, jar: Path, folder: Path, skipped: list[str]) -> Request:
    """The person's words as a request."""
    return Request(Season[season.upper()], JarPath(jar), PackFolder(folder), tuple(EffectName(name) for name in skipped))


@deal.has()
@deal.post(lambda result: set(seasons.COLOUR_MAPS) <= set(result) and set(markers.MARKED) <= set(result))
@deal.post(lambda result: cloudmap.CLOUD_MAP in result)
def wanted() -> tuple[TexturePath, ...]:
    """The textures a build asks the game for."""
    return (*seasons.COLOUR_MAPS, *markers.MARKED, cloudmap.CLOUD_MAP)


@deal.has()
# The game as vanilla reads it: every texture has the pixels its size says, which the seasons count on.
@deal.pre(lambda request, game: all(seasons.is_well_formed(texture) for texture in game.textures))
@deal.raises(seasons.MissingColourMap, cloudmap.MissingCloudMap, markers.MarkError, composer.ComposeError)
@deal.ensure(lambda _: _.result.vanilla == _.game.shaders and _.result.resource_format == _.game.resource_format)
@deal.ensure(lambda _: (clouds.NAME in _.request.skipped) != any(t.path == cloudmap.CLOUD_MAP for t in _.result.textures))
def planned(request: Request, game: Game) -> Build:
    """The pack for the request: the season's colours, the marked textures and the effects on the game's shaders."""
    colours = seasons.colour_maps(request.season, game.textures)
    marked = markers.mark(tuple(texture for texture in game.textures if texture.path in markers.MARKED))
    effects = tuple(made for made in (make() for make in EFFECTS) if made.name not in request.skipped)
    # The filled cloud map under the game's own cloud shaders is one slab of cloud over the whole sky, so
    # the map goes only where the clouds effect goes.
    cloud_map = () if clouds.NAME in request.skipped else (cloudmap.filled(game.textures),)
    textures = colours + marked + cloud_map
    return Build(game.resource_format, game.shaders, composer.compose(game.shaders, effects), textures)


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
        f"{request.pack}: {request.season.name.lower()}, game {game.version}, "
        f"{len(build.shaders)} shaders, {ended - started:.1f} s"
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
    parser.add_argument("--season", required=True, choices=season_names())
    parser.add_argument("--jar", type=Path, default=GAME_JAR)
    parser.add_argument("--pack", type=Path, default=PACK_FOLDER)
    parser.add_argument("--skip", action="append", default=[], choices=effect_names(), help="leave an effect out")
    arguments = parser.parse_args()
    return build(requested(arguments.season, arguments.jar, arguments.pack, arguments.skip))


if __name__ == "__main__":
    sys.exit(main())
