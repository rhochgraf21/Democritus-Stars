"""The game's own shaders and textures, as one Minecraft version ships them."""

import contextlib
import io
import json
import zipfile
import zlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NewType

import deal
from PIL import Image

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


# The path of the game's client jar, such as versions/26.3/26.3.jar in the game's folder.
JarPath = NewType("JarPath", Path)

# A folder under assets/minecraft/textures, ending in a slash, such as "block/".
TextureFolder = NewType("TextureFolder", str)

# The name of one entry inside a jar, from the jar's root, such as "version.json".
EntryName = NewType("EntryName", str)

# The bytes of one entry inside a jar, uncompressed.
EntryBytes = NewType("EntryBytes", bytes)

# A field of version.json, as the keys that lead to it joined by dots, such as "pack_version.resource_major".
VersionField = NewType("VersionField", str)

# The pixels of a decoded image, four bytes each in the order red, green, blue, alpha, row by row from the top left.
RgbaBytes = NewType("RgbaBytes", bytes)

_SHADERS = "assets/minecraft/shaders/"
_TEXTURES = "assets/minecraft/textures/"
# Only these are the shaders a pack can attach to: post/ holds post-processing chains that this pack leaves alone.
_SHADER_FOLDERS = ("core/", "include/")
_FOLDER_END = "/"  # how a zip names an entry that is a folder rather than a file
_TEXTURE_SUFFIX = ".png"
_META_SUFFIX = ".mcmeta"
_VERSION = EntryName("version.json")
_GAME_VERSION = VersionField("id")
_RESOURCE_MAJOR = VersionField("pack_version.resource_major")
_RESOURCE_MINOR = VersionField("pack_version.resource_minor")
_FIELD_SEPARATOR = "."
_TEXT_ENCODING = "utf-8"
_RGBA = "RGBA"
_CHANNELS = len(_RGBA)
# A corrupt deflate stream surfaces as zlib.error rather than as a zip error, and a missing or locked jar as an OSError.
_ZIP_FAILURES = (OSError, zipfile.BadZipFile, zlib.error)
# Pillow reports an unknown format or a truncated file as an OSError, but a few corrupt chunks escape its decoder as
# a SyntaxError or a ValueError: seen by corrupting a PNG at random, about one file in a thousand.
_PNG_FAILURES = (OSError, SyntaxError, ValueError)


class FieldKind(Enum):
    """The kind of value a field of version.json is expected to hold."""

    TEXT = "text"
    WHOLE_NUMBER = "a whole number"


class EntryFormat(Enum):
    """The format an entry of the jar is expected to be in."""

    TEXT = "UTF-8 text"
    JSON = "JSON"
    PNG = "a PNG image"


class VanillaError(Exception):
    """A jar that cannot be read as the client jar of one Minecraft version."""


# eq=False keeps an error hashable by identity, as Python's own errors are.
@dataclass(eq=False)
class JarUnreadable(VanillaError):
    """The jar is missing, or cannot be read as a zip file."""

    jar: JarPath

    def __str__(self) -> str:
        return f"{self.jar} cannot be read as a zip file"


@dataclass(eq=False)
class EntryAbsent(VanillaError):
    """Entries the game needs are not in the jar."""

    jar: JarPath
    entries: tuple[EntryName, ...]

    def __str__(self) -> str:
        return f"{self.jar} lacks {', '.join(self.entries)}"


@dataclass(eq=False)
class EntryUndecodable(VanillaError):
    """An entry of the jar is not in the format it is expected to be in."""

    jar: JarPath
    entry: EntryName
    expected: EntryFormat

    def __str__(self) -> str:
        return f"{self.entry} in {self.jar} is not {self.expected.value}"


@dataclass(eq=False)
class FieldAbsent(VanillaError):
    """version.json lacks a field, or holds a different kind of value there."""

    jar: JarPath
    field: VersionField
    expected: FieldKind

    def __str__(self) -> str:
        return f"{_VERSION} in {self.jar} has no {self.field} that is {self.expected.value}"


@dataclass(frozen=True)
class _JarIndex:
    """The names of every entry in one jar."""

    jar: JarPath
    names: frozenset[EntryName]


@dataclass(frozen=True)
class _JarEntries:
    """The entries read from one jar, by name."""

    jar: JarPath
    contents: Mapping[EntryName, EntryBytes]


@dataclass(frozen=True)
class _Release:
    """What version.json says about a game version."""

    version: GameVersion
    resource_format: ResourceFormat


@dataclass(frozen=True)
class _Image:
    """A decoded image, as its size and its pixels."""

    width: PixelCount
    height: PixelCount
    rgba: RgbaBytes


# ---- what the jar holds, decided without touching it -------------------------------------------------------------


@deal.has()
def _is_shader(name: EntryName) -> bool:
    """Whether an entry is a shader file that a pack can attach to."""
    return name.startswith(tuple(_SHADERS + folder for folder in _SHADER_FOLDERS)) and not name.endswith(_FOLDER_END)


@deal.has()
@deal.ensure(lambda names, result: all(_is_shader(n) and n in names for n in result))
@deal.ensure(lambda names, result: len(result) == sum(map(_is_shader, names)))
@deal.ensure(lambda names, result: list(result) == sorted(result))
def _shader_entries(names: frozenset[EntryName]) -> tuple[EntryName, ...]:
    """The shader files among entries, in order of name so that a build does not depend on the jar's order."""
    return tuple(sorted(n for n in names if _is_shader(n)))


@deal.has()
def _texture_entry(path: TexturePath) -> EntryName:
    """The entry that holds a texture."""
    return EntryName(_TEXTURES + path)


@deal.has()
def _meta_entry(path: TexturePath) -> EntryName:
    """The entry that holds a texture's animation, if the texture has one."""
    return EntryName(_TEXTURES + path + _META_SUFFIX)


@deal.has()
@deal.raises(EntryAbsent)
@deal.ensure(lambda index, wanted, result: all(n in index.names for n in result))
@deal.ensure(lambda index, wanted, result: _VERSION in result and all(_texture_entry(p) in result for p in wanted))
@deal.ensure(lambda index, wanted, result: set(_shader_entries(index.names)) <= set(result))
@deal.ensure(lambda index, wanted, result: all((_meta_entry(p) in result) == (_meta_entry(p) in index.names)
                                               for p in wanted))
def _reads(index: _JarIndex, wanted: tuple[TexturePath, ...]) -> tuple[EntryName, ...]:
    """The entries a game is made of: version.json, the shaders, and the wanted textures with any animation of theirs.

    Every missing entry is named at once, so that porting to a new version shows the whole gap in one run.
    """
    textures = tuple(_texture_entry(p) for p in wanted)
    missing = tuple(n for n in (_VERSION, *textures) if n not in index.names)
    if missing:
        raise EntryAbsent(index.jar, missing)
    metas = tuple(m for m in map(_meta_entry, wanted) if m in index.names)
    return (_VERSION, *_shader_entries(index.names), *textures, *metas)


@deal.has()
@deal.ensure(lambda names, folder, result: all(_texture_entry(p) in names for p in result))
@deal.ensure(lambda names, folder, result: all(p.startswith(folder) and p.endswith(_TEXTURE_SUFFIX) for p in result))
@deal.ensure(lambda names, folder, result: len(result) == sum(
    n.startswith(_TEXTURES + folder) and n.endswith(_TEXTURE_SUFFIX) for n in names))
@deal.ensure(lambda names, folder, result: list(result) == sorted(result))
def _textures_under(names: frozenset[EntryName], folder: TextureFolder) -> tuple[TexturePath, ...]:
    """The textures among entries that sit under a folder, at any depth, in order of path."""
    prefix = _TEXTURES + folder
    return tuple(sorted(TexturePath(n.removeprefix(_TEXTURES)) for n in names
                        if n.startswith(prefix) and n.endswith(_TEXTURE_SUFFIX)))


# ---- what the entries mean, decided without touching the jar -----------------------------------------------------


@deal.has()
@deal.raises(EntryUndecodable)
def _text(jar: JarPath, entry: EntryName, raw: EntryBytes) -> str:
    """An entry as text."""
    try:
        return raw.decode(_TEXT_ENCODING)
    except UnicodeDecodeError as error:
        raise EntryUndecodable(jar, entry, EntryFormat.TEXT) from error


@deal.has()
def _lookup(document: object, field: VersionField) -> object:
    """The value at a field of a JSON document, or None where the document has no such field."""
    value = document
    for key in field.split(_FIELD_SEPARATOR):
        value = value.get(key) if isinstance(value, dict) else None
    return value


@deal.has()
@deal.raises(FieldAbsent)
def _text_field(jar: JarPath, document: object, field: VersionField) -> str:
    """A field of version.json that holds text."""
    value = _lookup(document, field)
    if not isinstance(value, str):
        raise FieldAbsent(jar, field, FieldKind.TEXT)
    return value


@deal.has()
@deal.raises(FieldAbsent)
def _number_field(jar: JarPath, document: object, field: VersionField) -> int:
    """A field of version.json that holds a whole number."""
    value = _lookup(document, field)
    # JSON's true and false arrive as Python bools, which are ints, yet no version number is written that way.
    if not isinstance(value, int) or isinstance(value, bool):
        raise FieldAbsent(jar, field, FieldKind.WHOLE_NUMBER)
    return value


@deal.has()
@deal.raises(EntryUndecodable, FieldAbsent)
def _release(jar: JarPath, raw: EntryBytes) -> _Release:
    """The game version and resource pack format that version.json names."""
    try:
        document = json.loads(_text(jar, _VERSION, raw))
    except json.JSONDecodeError as error:
        raise EntryUndecodable(jar, _VERSION, EntryFormat.JSON) from error
    major = FormatNumber(_number_field(jar, document, _RESOURCE_MAJOR))
    minor = FormatNumber(_number_field(jar, document, _RESOURCE_MINOR))
    return _Release(GameVersion(_text_field(jar, document, _GAME_VERSION)), ResourceFormat(major, minor))


@deal.has()
@deal.raises(EntryUndecodable)
def _decoded(jar: JarPath, entry: EntryName, raw: EntryBytes) -> _Image:
    """A PNG entry decoded to RGBA, whatever mode it is stored in."""
    try:
        with Image.open(io.BytesIO(raw)) as stored:
            image = stored.convert(_RGBA)
    except _PNG_FAILURES as error:
        raise EntryUndecodable(jar, entry, EntryFormat.PNG) from error
    width, height = image.size
    return _Image(PixelCount(width), PixelCount(height), RgbaBytes(image.tobytes()))


@deal.has()
@deal.pre(lambda rgba: len(rgba) % _CHANNELS == 0)
@deal.ensure(lambda rgba, result: len(result) * _CHANNELS == len(rgba))
@deal.ensure(lambda rgba, result: all(
    (p.red, p.green, p.blue, p.alpha) == tuple(rgba[i * _CHANNELS:(i + 1) * _CHANNELS]) for i, p in enumerate(result)))
def _pixels(rgba: RgbaBytes) -> tuple[Pixel, ...]:
    """Pixels from RGBA bytes, in the order the bytes hold them."""
    return tuple(Pixel(*(Channel(c) for c in rgba[start:start + _CHANNELS])) for start in range(0, len(rgba), _CHANNELS))


@deal.has()
@deal.raises(EntryUndecodable)
def _texture(entries: _JarEntries, path: TexturePath) -> Texture:
    """One wanted texture, decoded, with its animation if the jar holds one beside it."""
    entry = _texture_entry(path)
    image = _decoded(entries.jar, entry, entries.contents[entry])
    meta_entry = _meta_entry(path)
    meta = entries.contents.get(meta_entry)
    return Texture(path, image.width, image.height, _pixels(image.rgba),
                   None if meta is None else TextureMeta(_text(entries.jar, meta_entry, meta)))


@deal.has()
@deal.raises(EntryUndecodable)
def _shader(entries: _JarEntries, entry: EntryName) -> ShaderFile:
    """One shader file, named by where it sits under the shaders folder."""
    text = _text(entries.jar, entry, entries.contents[entry])
    return ShaderFile(ShaderPath(entry.removeprefix(_SHADERS)), ShaderText(text))


@deal.has()
@deal.raises(EntryUndecodable, FieldAbsent)
@deal.pre(lambda entries, wanted: _VERSION in entries.contents and all(
    _texture_entry(p) in entries.contents for p in wanted))
@deal.ensure(lambda entries, wanted, result: tuple(t.path for t in result.textures) == wanted)
@deal.ensure(lambda entries, wanted, result: tuple(_SHADERS + s.path for s in result.shaders) == _shader_entries(
    frozenset(entries.contents)))
def _game(entries: _JarEntries, wanted: tuple[TexturePath, ...]) -> Game:
    """The game that the entries read from a jar make up."""
    release = _release(entries.jar, entries.contents[_VERSION])
    shaders = tuple(_shader(entries, n) for n in _shader_entries(frozenset(entries.contents)))
    textures = tuple(_texture(entries, p) for p in wanted)
    return Game(release.version, release.resource_format, shaders, textures)


# ---- the jar itself ----------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _opened(jar: JarPath) -> Iterator[zipfile.ZipFile]:
    """The jar, open for reading, with any failure to read it reported as the jar being unreadable."""
    try:
        with zipfile.ZipFile(jar) as archive:
            yield archive
    except _ZIP_FAILURES as error:
        raise JarUnreadable(jar) from error


@deal.raises(JarUnreadable, EntryAbsent, EntryUndecodable, FieldAbsent)
def read_game(jar: JarPath, wanted: tuple[TexturePath, ...]) -> Game:
    """The version, the shaders under core/ and include/, and the wanted textures of the game in a client jar."""
    with _opened(jar) as archive:
        index = _JarIndex(jar, frozenset(EntryName(name) for name in archive.namelist()))
        entries = _JarEntries(jar, {entry: EntryBytes(archive.read(entry)) for entry in _reads(index, wanted)})
    return _game(entries, wanted)


@deal.raises(JarUnreadable)
def texture_paths(jar: JarPath, folder: TextureFolder) -> tuple[TexturePath, ...]:
    """Every texture a client jar holds under a folder of the textures, at any depth, in order of path."""
    with _opened(jar) as archive:
        names = frozenset(EntryName(n) for n in archive.namelist())
    return _textures_under(names, folder)
