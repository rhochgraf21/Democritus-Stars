"""What the tests share: the real game, read once.

An effect is code for a line of the game's own shader text, so the only honest test of it is against that
text. The game's files are Mojang's and are not kept in this repository; the tests read them from the
player's installed game, or from the jar that DEMOCRITUS_GAME_JAR names.
"""

import os
from pathlib import Path

import pytest

from democritus.command import GAME_JAR
from democritus.vanilla import Game, JarPath, read_game

JAR_VARIABLE = "DEMOCRITUS_GAME_JAR"


@pytest.fixture(scope="session")
def game() -> Game:
    """The shaders of the installed game. A test that needs them fails, and is not skipped, without it."""
    jar = Path(os.environ.get(JAR_VARIABLE, GAME_JAR))
    if not jar.is_file():
        # Failing keeps a missing game from passing for a tested effect: a skipped test proves nothing.
        pytest.fail(f"no game jar at {jar}: install Minecraft 26.3 or set {JAR_VARIABLE}")
    return read_game(JarPath(jar), ())
