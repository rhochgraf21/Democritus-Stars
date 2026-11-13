"""Each change to the look, as GLSL code with the vanilla shader that the code attaches to."""

from dataclasses import dataclass
from enum import Enum
from typing import NewType

from shader.vanilla import ShaderFile, ShaderPath

# What an effect is called when a build has to name it, such as "fog".
EffectName = NewType("EffectName", str)

# GLSL lines that an effect adds to a vanilla shader.
GlslCode = NewType("GlslCode", str)

# One line of a vanilla shader's own text, without the space around it, that marks where code goes.
#
# A line of the game's text, and not a line number, so that a new game version which moves the line
# changes nothing, and one which removes it is noticed.
Place = NewType("Place", str)


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
