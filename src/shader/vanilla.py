"""The game's own shaders and textures, as one Minecraft version ships them."""

from dataclasses import dataclass
from typing import NewType

# The name Mojang gives a release, such as "26.3".
GameVersion = NewType("GameVersion", str)

# One number of the resource pack format a game version accepts.
FormatNumber = NewType("FormatNumber", int)

# Where a shader file sits under assets/minecraft/shaders, such as "core/terrain.fsh".
ShaderPath = NewType("ShaderPath", str)

# The whole GLSL text of one shader file.
ShaderText = NewType("ShaderText", str)

# Where a texture sits under assets/minecraft/textures, such as "block/water_still.png".
TexturePath = NewType("TexturePath", str)

# The text of the .mcmeta file beside a texture, which holds its animation.
TextureMeta = NewType("TextureMeta", str)

# One colour channel of a pixel, from 0 to 255.
Channel = NewType("Channel", int)

# A width or a height of a texture, in pixels.
PixelCount = NewType("PixelCount", int)


@dataclass(frozen=True)
class ResourceFormat:
    """The resource pack format a game version accepts, as its major and minor number."""

    major: FormatNumber
    minor: FormatNumber


@dataclass(frozen=True)
class ShaderFile:
    """One shader file: where it sits, and its text."""

    path: ShaderPath
    text: ShaderText


@dataclass(frozen=True)
class Pixel:
    """One pixel of a texture."""

    red: Channel
    green: Channel
    blue: Channel
    alpha: Channel


@dataclass(frozen=True)
class Texture:
    """One texture, decoded: its pixels row by row from the top left, and its animation if it has one.

    The game takes a texture's .mcmeta from the same pack as the texture, so a pack that changes an
    animated texture has to carry the animation along or the texture stops moving.
    """

    path: TexturePath
    width: PixelCount
    height: PixelCount
    pixels: tuple[Pixel, ...]
    meta: TextureMeta | None


@dataclass(frozen=True)
class Game:
    """What one Minecraft version ships that a pack builds on."""

    version: GameVersion
    resource_format: ResourceFormat
    shaders: tuple[ShaderFile, ...]
    textures: tuple[Texture, ...]
