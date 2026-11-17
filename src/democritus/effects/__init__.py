"""Each change to the look, as GLSL code with the vanilla shader that the code attaches to."""

from dataclasses import dataclass
from enum import Enum
from typing import NewType

from democritus.vanilla import ShaderFile, ShaderPath

# What an effect is called when a build has to name it, such as "fog".
EffectName = NewType("EffectName", str)

# GLSL lines that an effect adds to a vanilla shader.
GlslCode = NewType("GlslCode", str)

# One line of a vanilla shader's own text, without the space around it, that marks where code goes.
#
# A line of the game's text, and not a line number, so that a new game version which moves the line
# changes nothing, and one which removes it is noticed.
ExactLine = NewType("ExactLine", str)


class Landmark(Enum):
    """A place that every shader has by its shape, found by what a line is and not by its exact text.

    Code that only needs to be somewhere in the shader's structure, such as a new varying after the others,
    attaches to a landmark, and so survives a game version that rewords the lines around it.
    """

    # Each value is what an error calls the landmark: "core/sky.fsh has no input".
    LAST_INCLUDE = "include"
    LAST_INPUT = "input"
    LAST_OUTPUT = "output"
    MAIN = "main"
    END_OF_MAIN = "end of main"


# Where an effect's code goes. An exact line only where the code depends on what that line does, such as a
# statement it replaces: if Mojang changes that statement, the code may be wrong, and the build should stop.
# A landmark everywhere else.
Place = ExactLine | Landmark


class Position(Enum):
    """Where an effect's code goes, relative to its place."""

    BEFORE = "before"
    AFTER = "after"
    INSTEAD = "instead"


@dataclass(frozen=True)
class Attachment:
    """Code for one vanilla shader, and the place in that shader where the code goes."""

    shader: ShaderPath
    place: Place
    position: Position
    code: GlslCode


@dataclass(frozen=True)
class Effect:
    """One change to the look: its attachments, and the new include files that their code uses."""

    name: EffectName
    attachments: tuple[Attachment, ...]
    includes: tuple[ShaderFile, ...] = ()
