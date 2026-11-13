"""The one command that builds the pack for a season.

A first version, written to get a pack into the game: it has no tests and none of the split between
deciding and doing that the other parts have. It is to be rebuilt as a proper part.
"""

import argparse
import sys
import time
from pathlib import Path

from shader import composer, markers, pack, seasons, validator, vanilla
from shader.effects import clouds, lighting, sky, stars, weather

MINECRAFT = Path.home() / "Library/Application Support/minecraft"
GAME_JAR = MINECRAFT / "versions/26.3/26.3.jar"
PACK_FOLDER = MINECRAFT / "resourcepacks/zigdag-shaders"

# Order matters where two effects put code at the same place of one shader: it stacks in this order.
EFFECTS = {
    "lighting": lighting.effect,
    "sky": sky.effect,
    "stars": stars.effect,
    "clouds": clouds.effect,
    "weather": weather.effect,
}


def main() -> int:
    """Build the pack, and say where it went or why it was not written."""
    parser = argparse.ArgumentParser(prog="shader", description=__doc__)
    parser.add_argument("--season", required=True, choices=[season.name.lower() for season in seasons.Season])
    parser.add_argument("--jar", type=Path, default=GAME_JAR)
    parser.add_argument("--pack", type=Path, default=PACK_FOLDER)
    parser.add_argument("--skip", action="append", default=[], choices=list(EFFECTS), help="leave an effect out")
    arguments = parser.parse_args()
    started = time.perf_counter()

    season = seasons.Season[arguments.season.upper()]
    game = vanilla.read_game(vanilla.JarPath(arguments.jar), seasons.COLOUR_MAPS + markers.MARKED)
    colour_maps = seasons.colour_maps(season, game.textures)
    marked = markers.mark(tuple(texture for texture in game.textures if texture.path in markers.MARKED))
    effects = tuple(make() for name, make in EFFECTS.items() if name not in arguments.skip)
    built = composer.compose(game.shaders, effects)

    failures = validator.validate(built, game.shaders)
    if failures:
        # One shader that does not compile makes the game switch the whole pack off, so nothing is written.
        for failure in failures:
            print(f"{failure.shader} {failure.defines}:\n{failure.message}", file=sys.stderr)
        print(f"not written: {len(failures)} shaders do not compile", file=sys.stderr)
        return 1

    pack.write_pack(pack.PackFolder(arguments.pack), game.resource_format, built, colour_maps + marked)
    seconds = time.perf_counter() - started
    print(f"{arguments.pack}: {season.name.lower()}, game {game.version}, {len(built)} shaders, {seconds:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
