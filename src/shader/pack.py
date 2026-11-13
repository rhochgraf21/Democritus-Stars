"""Writes the built shaders and textures as a resource pack in the resource pack folder of the game.

Everything the pack will hold, and whether its folder may be cleared first, is decided by pure functions that
return a plan; `write_pack` only reads the folder, asks for the plan and carries it out.
"""

import io
import json
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import NewType

import deal
from PIL import Image

from shader.vanilla import Channel, PixelCount, ResourceFormat, ShaderFile, ShaderPath, Texture, TexturePath

# The folder of one resource pack, such as .../minecraft/resourcepacks/<pack name>.
PackFolder = NewType("PackFolder", Path)

# The bytes of one file of the pack, exactly as they are written.
FileContent = NewType("FileContent", bytes)

# The text of pack.mcmeta, which tells the game what the folder is and which formats it supports.
PackMetaText = NewType("PackMetaText", str)

PACK_META = PurePosixPath("pack.mcmeta")
SHADERS = PurePosixPath("assets/minecraft/shaders")
TEXTURES = PurePosixPath("assets/minecraft/textures")
TEXTURE_META_SUFFIX = ".mcmeta"
DESCRIPTION = "Shaders built on the game's own"
ENCODING = "utf-8"
PNG = "PNG"
RGBA = "RGBA"
PARENT = ".."
SMALLEST_SIDE = PixelCount(1)
DARKEST = Channel(0)
BRIGHTEST = Channel(255)

# Any resource pack has a pack.mcmeta, so that file cannot tell a pack of ours from one a person made by hand.
# This file is written by no one else, and it says to a person who opens the folder why it may vanish.
MARKER = PurePosixPath("built-by-shader.txt")
MARKER_TEXT = (
    "This pack is written by the shader build. Each build deletes this folder and writes it anew, "
    "so a change made here by hand is lost.\n"
)
OWN_FILES = (PACK_META, MARKER)  # the files every pack holds, whatever it is built from


class NotOurPack(Exception):
    """The pack folder holds files that this tool did not write, so a build will not clear it."""

    def __init__(self, folder: PackFolder) -> None:
        super().__init__(f"{folder} is not a pack that this tool wrote; move it away, or choose another folder")
        self.folder = folder


class Clearance(Enum):
    """What a build may do with what is already in the pack folder."""

    NOTHING_TO_CLEAR = "nothing to clear"
    CLEAR = "clear"
    REFUSE = "refuse"


@dataclass(frozen=True)
class FolderState:
    """What is at the pack folder before a build: what a decision to clear it rests on."""

    exists: bool
    is_folder: bool
    entries: tuple[Path, ...]
    marked: bool


@dataclass(frozen=True)
class PlannedFile:
    """One file the pack will hold: where it goes and its bytes."""

    path: Path
    content: FileContent


@dataclass(frozen=True)
class PackPlan:
    """Everything a build writes, and what it does first with what is already in the folder."""

    folder: PackFolder
    clearance: Clearance
    files: tuple[PlannedFile, ...]


@deal.pure
@deal.ensure(lambda folder, result: result.parent == folder)
def marker_of(folder: PackFolder) -> Path:
    """Where the file sits that marks a pack folder as written by this tool."""
    return folder / MARKER


@deal.pure
@deal.ensure(lambda state, result: result is not Clearance.CLEAR or state.marked)
def clearance(state: FolderState) -> Clearance:
    """What a build may do with the pack folder: clear it if this tool wrote it, refuse it if it holds anything else."""
    # A rebuild has to delete what an earlier build left, or a removed effect lingers in the game. Deleting is
    # only safe where nothing could be lost: a folder that holds anything not of our making is left alone.
    if not state.exists:
        return Clearance.NOTHING_TO_CLEAR
    if not state.is_folder:
        return Clearance.REFUSE
    if state.marked:
        return Clearance.CLEAR
    if not state.entries:
        return Clearance.NOTHING_TO_CLEAR
    return Clearance.REFUSE


@deal.pure
@deal.ensure(lambda resource_format, result: json.loads(result)["pack"]["min_format"]
             == [resource_format.major, resource_format.minor])
def pack_meta(resource_format: ResourceFormat) -> PackMetaText:
    """The text of pack.mcmeta for a pack that supports exactly one resource format."""
    # The shape since format 65: min_format and max_format as [major, minor], with pack_format no longer
    # required. It is written from memory, not checked against the game, so it is kept here in one place.
    version = [resource_format.major, resource_format.minor]
    return PackMetaText(json.dumps({"pack": {"description": DESCRIPTION, "min_format": version,
                                             "max_format": version}}))


@deal.pure
def _rgba(texture: Texture) -> tuple[tuple[Channel, Channel, Channel, Channel], ...]:
    """A texture's pixels as Pillow takes and gives them in RGBA."""
    return tuple((p.red, p.green, p.blue, p.alpha) for p in texture.pixels)


@deal.pure
def _drawable(texture: Texture) -> bool:
    """Whether a texture can be encoded: at least one pixel each way, one pixel per place, each channel in range."""
    return (texture.width >= SMALLEST_SIDE and texture.height >= SMALLEST_SIDE
            and len(texture.pixels) == texture.width * texture.height
            and all(DARKEST <= c <= BRIGHTEST for p in texture.pixels for c in (p.red, p.green, p.blue, p.alpha)))


@deal.pure
@deal.pre(_drawable)
@deal.ensure(lambda texture, result: Image.open(io.BytesIO(result)).convert(RGBA).get_flattened_data()
             == _rgba(texture))
def png_of(texture: Texture) -> FileContent:
    """A texture encoded as an RGBA PNG."""
    image = Image.new(RGBA, (texture.width, texture.height))
    image.putdata(_rgba(texture))
    out = io.BytesIO()
    image.save(out, format=PNG)
    return FileContent(out.getvalue())


@deal.pure
def _stays_inside(path: ShaderPath | TexturePath) -> bool:
    """Whether a path relative to the pack stays inside it: a file written outside could be anything of the user's."""
    relative = PurePosixPath(path)
    return not relative.is_absolute() and PARENT not in relative.parts


@deal.pure
def _texture_files(folder: PackFolder, texture: Texture) -> tuple[PlannedFile, ...]:
    """The PNG of a texture and, when it is animated, its .mcmeta beside it."""
    # The game takes a texture's animation from the same pack as the texture, so dropping the .mcmeta
    # freezes an animated texture.
    png = PlannedFile(folder / TEXTURES / texture.path, png_of(texture))
    if texture.meta is None:
        return (png,)
    meta = PlannedFile(folder / TEXTURES / f"{texture.path}{TEXTURE_META_SUFFIX}",
                       FileContent(texture.meta.encode(ENCODING)))
    return png, meta


@deal.pure
@deal.pre(lambda folder, resource_format, shaders, textures, state: all(
    _stays_inside(f.path) for f in (*shaders, *textures)))
@deal.pre(lambda folder, resource_format, shaders, textures, state: all(_drawable(t) for t in textures))
@deal.ensure(lambda folder, resource_format, shaders, textures, state, result: len(result.files) == len(shaders)
             + len(textures) + sum(t.meta is not None for t in textures) + len(OWN_FILES))
@deal.ensure(lambda folder, resource_format, shaders, textures, state, result: all(
    f.path.is_relative_to(folder) for f in result.files))
def plan_pack(folder: PackFolder, resource_format: ResourceFormat, shaders: tuple[ShaderFile, ...],
              textures: tuple[Texture, ...], state: FolderState) -> PackPlan:
    """Every file of the pack with its bytes, and what to do first with what is already in its folder."""
    own = (PlannedFile(folder / PACK_META, FileContent(pack_meta(resource_format).encode(ENCODING))),
           PlannedFile(marker_of(folder), FileContent(MARKER_TEXT.encode(ENCODING))))
    built = tuple(PlannedFile(folder / SHADERS / s.path, FileContent(s.text.encode(ENCODING))) for s in shaders)
    drawn = tuple(f for t in textures for f in _texture_files(folder, t))
    return PackPlan(folder, clearance(state), own + built + drawn)


def _survey(folder: PackFolder) -> FolderState:
    """What is at the pack folder now."""
    return FolderState(exists=folder.exists(), is_folder=folder.is_dir(), entries=tuple(folder.glob("*")),
                       marked=marker_of(folder).is_file())


def _carry_out(plan: PackPlan) -> None:
    """Clears the folder if the plan says so, then writes each file of the plan."""
    match plan.clearance:
        case Clearance.REFUSE:
            raise NotOurPack(plan.folder)
        case Clearance.CLEAR:
            shutil.rmtree(plan.folder)
        case Clearance.NOTHING_TO_CLEAR:
            pass
    for file in plan.files:
        file.path.parent.mkdir(parents=True, exist_ok=True)
        file.path.write_bytes(file.content)


@deal.raises(NotOurPack)
def write_pack(folder: PackFolder, resource_format: ResourceFormat, shaders: tuple[ShaderFile, ...],
               textures: tuple[Texture, ...]) -> None:
    """Writes the shaders and textures as a resource pack in `folder`, replacing a pack this tool wrote there before.

    Raises NotOurPack, and leaves the folder as it was, when the folder holds anything this tool did not write.
    """
    _carry_out(plan_pack(folder, resource_format, shaders, textures, _survey(folder)))
