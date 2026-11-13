"""Joins the effects onto the vanilla shaders, at the places in the game's own text that they name.

A place is matched against the original text of its shader, never against text that another effect has
already changed: each effect then means the same thing whichever effects it is built with, and a place
that a new game version removes or repeats is named, with the effect, rather than guessed at.
"""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import NewType

import deal

from shader.effects import Attachment, Effect, EffectName, GlslCode, Place, Position
from shader.vanilla import ShaderFile, ShaderPath, ShaderText

# The text of one line of a shader, without its line ending.
_LineBody = NewType("LineBody", str)

# Which line of a shader's original text, counted from 0.
_LineIndex = NewType("LineIndex", int)


class _LineEnding(Enum):
    """What ends one line of a shader's text."""

    LF = "\n"
    CRLF = "\r\n"
    # Only the last line of a text can lack an ending.
    NONE = ""


# Code with no line ending of its own is given this one, when its shader has none to copy either.
_DEFAULT_ENDING = _LineEnding.LF

# What CRLF holds before its LF.
_CARRIAGE_RETURN = _LineEnding.CRLF.value.removesuffix(_LineEnding.LF.value)


@dataclass(frozen=True)
class _Line:
    """One line of a shader's text, with the ending it had, so that the text can be put back byte for byte."""

    body: _LineBody
    ending: _LineEnding

    @property
    def text(self) -> ShaderText:
        """The line as it stands in the shader's text."""
        return ShaderText(self.body + self.ending.value)

    def holds(self, place: Place) -> bool:
        """Whether this line is the place: the same text, once the space around it is stripped."""
        return self.body.strip() == place

    def ended_with(self, ending: _LineEnding) -> "_Line":
        """This line, given an ending if it has none, so that code can follow it."""
        if self.ending is _LineEnding.NONE:
            return _Line(self.body, ending)
        return self


class ComposeError(Exception):
    """An effect that cannot be joined onto the vanilla shaders as it is written."""


@dataclass
class PlaceMissing(ComposeError):
    """An effect attaches to a place that its vanilla shader lacks."""

    effect: EffectName
    shader: ShaderPath
    place: Place

    def __str__(self) -> str:
        return f"effect {self.effect}: {self.shader} has no line {self.place!r}"


@dataclass
class PlaceRepeated(ComposeError):
    """An effect attaches to a place that its vanilla shader holds more than once."""

    effect: EffectName
    shader: ShaderPath
    place: Place

    def __str__(self) -> str:
        return f"effect {self.effect}: {self.shader} has the line {self.place!r} more than once"


@dataclass
class ShaderUnknown(ComposeError):
    """An effect attaches to a shader that the game does not ship."""

    effect: EffectName
    shader: ShaderPath
    place: Place

    def __str__(self) -> str:
        return f"effect {self.effect}: no vanilla shader {self.shader} to attach {self.place!r} to"


@dataclass
class ReplacementsConflict(ComposeError):
    """Two effects both put their code instead of the same place."""

    effect: EffectName
    other: EffectName
    shader: ShaderPath
    place: Place

    def __str__(self) -> str:
        return f"effects {self.effect} and {self.other} both replace {self.place!r} in {self.shader}"


@dataclass
class IncludesConflict(ComposeError):
    """Two effects add an include file at the same path with different text."""

    effect: EffectName
    other: EffectName
    shader: ShaderPath

    def __str__(self) -> str:
        return f"effects {self.effect} and {self.other} add different files at {self.shader}"


@dataclass
class IncludeReplacesVanilla(ComposeError):
    """An effect adds an include file at the path of a vanilla shader."""

    effect: EffectName
    shader: ShaderPath

    def __str__(self) -> str:
        return f"effect {self.effect} adds {self.shader}, which would replace the vanilla shader"


@dataclass(frozen=True)
class _Placed:
    """One attachment, with the effect it belongs to, so that an error can name the effect."""

    effect: EffectName
    attachment: Attachment


@dataclass(frozen=True)
class _Added:
    """One include file, with the effect that adds it, so that an error can name the effect."""

    effect: EffectName
    file: ShaderFile


@dataclass(frozen=True)
class _Edit:
    """All the code that goes at one place: before it, after it, and instead of it."""

    before: tuple[GlslCode, ...] = ()
    after: tuple[GlslCode, ...] = ()
    instead: _Placed | None = None


@deal.has()
@deal.ensure(lambda text, result: "".join(line.text for line in result) == text)
def _split_lines(text: ShaderText) -> tuple[_Line, ...]:
    """The lines of a shader's text, each with its own ending."""
    # Split on LF alone: str.splitlines also splits on form feeds and Unicode separators, which GLSL
    # does not treat as line ends, so a place could match text the compiler sees as part of a longer line.
    pieces = text.split(_LineEnding.LF.value)
    ended = tuple(_ended_line(piece) for piece in pieces[:-1])
    last = pieces[-1]
    if last:
        return (*ended, _Line(_LineBody(last), _LineEnding.NONE))
    return ended


@deal.has()
def _ended_line(piece: str) -> _Line:
    if piece.endswith(_CARRIAGE_RETURN):
        return _Line(_LineBody(piece.removesuffix(_CARRIAGE_RETURN)), _LineEnding.CRLF)
    return _Line(_LineBody(piece), _LineEnding.LF)


@deal.has()
def _code_lines(code: GlslCode, ending: _LineEnding) -> tuple[_Line, ...]:
    """An effect's code as lines of its shader, all ending the way the shader's lines do."""
    return tuple(_Line(line.body, ending) for line in _split_lines(ShaderText(code)))


@deal.has()
def _ending_for(place: _Line, lines: tuple[_Line, ...]) -> _LineEnding:
    """The ending to give code at a place: the place's own, else the first the shader uses."""
    endings = (line.ending for line in (place, *lines) if line.ending is not _LineEnding.NONE)
    return next(endings, _DEFAULT_ENDING)


@deal.has()
@deal.raises(PlaceMissing, PlaceRepeated)
def _find_place(shader: ShaderFile, lines: tuple[_Line, ...], placed: _Placed) -> _LineIndex:
    """Which line of the shader is the place that an attachment names."""
    place = placed.attachment.place
    found = [_LineIndex(index) for index, line in enumerate(lines) if line.holds(place)]
    if not found:
        raise PlaceMissing(placed.effect, shader.path, place)
    if len(found) > 1:
        raise PlaceRepeated(placed.effect, shader.path, place)
    return found[0]


@deal.has()
@deal.raises(ReplacementsConflict)
def _with_attachment(edit: _Edit, placed: _Placed, shader: ShaderPath) -> _Edit:
    """The edit at a place, with one more attachment's code."""
    code = placed.attachment.code
    match placed.attachment.position:
        case Position.BEFORE:
            return _Edit((*edit.before, code), edit.after, edit.instead)
        case Position.AFTER:
            return _Edit(edit.before, (*edit.after, code), edit.instead)
        case Position.INSTEAD:
            if edit.instead is not None:
                raise ReplacementsConflict(edit.instead.effect, placed.effect, shader, placed.attachment.place)
            return _Edit(edit.before, edit.after, placed)


@deal.has()
def _edited_lines(line: _Line, edit: _Edit, ending: _LineEnding) -> tuple[_Line, ...]:
    """One line of the original text, with the code that goes before, after or instead of it."""
    before = tuple(new for code in edit.before for new in _code_lines(code, ending))
    after = tuple(new for code in edit.after for new in _code_lines(code, ending))
    if edit.instead is not None:
        middle = _code_lines(edit.instead.attachment.code, ending)
    elif after:
        middle = (line.ended_with(ending),)
    else:
        middle = (line,)
    return (*before, *middle, *after)


@deal.has()
@deal.raises(PlaceMissing, PlaceRepeated, ReplacementsConflict)
def _join_shader(shader: ShaderFile, placements: tuple[_Placed, ...]) -> ShaderFile:
    """One vanilla shader, with the code of each attachment to it at its place."""
    lines = _split_lines(shader.text)
    edits: dict[_LineIndex, _Edit] = defaultdict(_Edit)
    for placed in placements:
        index = _find_place(shader, lines, placed)
        edits[index] = _with_attachment(edits[index], placed, shader.path)
    joined = (
        new
        for index, line in enumerate(lines)
        for new in (
            _edited_lines(line, edits[_LineIndex(index)], _ending_for(line, lines))
            if _LineIndex(index) in edits
            else (line,)
        )
    )
    return ShaderFile(shader.path, ShaderText("".join(line.text for line in joined)))


@deal.has()
@deal.raises(ShaderUnknown)
def _placements_by_shader(
    vanilla: tuple[ShaderFile, ...], effects: tuple[Effect, ...]
) -> dict[ShaderPath, tuple[_Placed, ...]]:
    """Every attachment of the effects, grouped by its vanilla shader, in the order the effects are given."""
    known = {shader.path for shader in vanilla}
    grouped: dict[ShaderPath, list[_Placed]] = defaultdict(list)
    for effect in effects:
        for attachment in effect.attachments:
            if attachment.shader not in known:
                raise ShaderUnknown(effect.name, attachment.shader, attachment.place)
            grouped[attachment.shader].append(_Placed(effect.name, attachment))
    return {path: tuple(placements) for path, placements in grouped.items()}


@deal.has()
@deal.raises(IncludesConflict, IncludeReplacesVanilla)
def _collect_includes(vanilla: tuple[ShaderFile, ...], effects: tuple[Effect, ...]) -> tuple[ShaderFile, ...]:
    """Every include file the effects add, once each."""
    known = {shader.path for shader in vanilla}
    added: dict[ShaderPath, _Added] = {}
    for effect in effects:
        for include in effect.includes:
            if include.path in known:
                raise IncludeReplacesVanilla(effect.name, include.path)
            first = added.setdefault(include.path, _Added(effect.name, include))
            if first.file.text != include.text:
                raise IncludesConflict(first.effect, effect.name, include.path)
    return tuple(entry.file for entry in added.values())


def _paths(files: tuple[ShaderFile, ...]) -> list[ShaderPath]:
    return [file.path for file in files]


def _attached_paths(effects: tuple[Effect, ...]) -> set[ShaderPath]:
    return {attachment.shader for effect in effects for attachment in effect.attachments}


def _include_paths(effects: tuple[Effect, ...]) -> set[ShaderPath]:
    return {include.path for effect in effects for include in effect.includes}


@deal.has()
@deal.raises(
    PlaceMissing, PlaceRepeated, ShaderUnknown, ReplacementsConflict, IncludesConflict, IncludeReplacesVanilla
)
@deal.post(lambda result: _paths(result) == sorted(set(_paths(result))))
@deal.ensure(
    lambda vanilla, effects, result: set(_paths(result)) == _attached_paths(effects) | _include_paths(effects)
)
def compose(vanilla: tuple[ShaderFile, ...], effects: tuple[Effect, ...]) -> tuple[ShaderFile, ...]:
    """The pack's shader files: each vanilla shader the effects attach to, joined, and each include they add.

    Vanilla shaders that nothing attaches to are left out, since the game falls back to its own. The files
    come sorted by path, so the same effects build the same pack whatever order the shaders are read in.
    """
    placements = _placements_by_shader(vanilla, effects)
    includes = _collect_includes(vanilla, effects)
    joined = tuple(_join_shader(shader, placements[shader.path]) for shader in vanilla if shader.path in placements)
    return tuple(sorted((*joined, *includes), key=lambda file: file.path))
