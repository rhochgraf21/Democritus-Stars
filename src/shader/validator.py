"""Compiles each shader the pack changes, under each combination of the defines it tests, before the pack is written.

One shader error and the game switches the whole pack off, so a shader is only safe when every variant the game
may compile of it compiles. Which variants the game really compiles is written in the game's code, not in its files,
so every combination of the names a shader tests is compiled instead. A combination under which the untouched vanilla
shader fails too is not one the game uses, and is not held against the pack.

Everything here that decides is a calculation; only running the compiler is not.
"""

import itertools
import operator
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import NewType, TypeAlias

import deal

from shader.vanilla import ShaderFile, ShaderPath, ShaderText

# A name the GLSL preprocessor knows, such as "OIT_ALPHA_ONLY".
MacroName = NewType("MacroName", str)

# What a defined name stands for, such as "0.1"; empty for a name that is only switched on.
MacroValue = NewType("MacroValue", str)

# The names a shader was compiled with switched on, in alphabetical order.
Defines: TypeAlias = tuple[MacroName, ...]

# What the compiler said about one shader, with each place in it given as a file of the pack and a line.
CompilerMessage = NewType("CompilerMessage", str)

# A line number in a shader file, counted from 1.
LineNumber = NewType("LineNumber", int)

# One line of GLSL text, without its line break.
GlslLine = NewType("GlslLine", str)

# One word of the command line that runs the compiler.
Argument = NewType("Argument", str)

# What a finished compiler run exits with.
ExitCode = NewType("ExitCode", int)

# Everything a compiler run printed.
CompilerOutput = NewType("CompilerOutput", str)

# The reference GLSL compiler. The game compiles with shaderc, which is glslang inside.
COMPILER = Argument("glslangValidator")

# The options that 26.3's GlslCompiler gives shaderc: Vulkan 1.2 rules, and a binding for each uniform that names
# none. Under these, every untouched vanilla core shader compiles under some combination of its defines. Under
# OpenGL rules none does, and under Vulkan rules without the bindings all but one fail.
_RULES = (Argument("--target-env"), Argument("vulkan1.2"), Argument("--auto-map-bindings"))

# The shader text goes in on standard input, so nothing is written to disk.
_FROM_INPUT = (Argument("--stdin"), Argument("-S"))

# Only the verdict is wanted, not the SPIR-V.
_NO_BINARY = (Argument("-o"), Argument(os.devnull))

_DEFINE = "-D"  # the compiler option that defines a name
_SUCCESS = ExitCode(0)

# What glslang calls standard input in its messages, which open with that name on a line of its own.
_INPUT_NAME = "stdin"
_INPUT_PLACE = re.compile(rf"\b{_INPUT_NAME}:(?P<line>\d+):")
_END = "end"  # the place of a message about the end of the text, past its last line

# Values that the game gives along with a shader's defines, and that no shader file tests for, read from 26.3's
# RenderPipelines and LevelRenderer. The shaders use them as numbers, for an array size or a loop bound, so without
# them rendertype_end_portal.fsh and every OIT variant fails even untouched. They are the game's code, not its files,
# so they change only with a new version of the game.
_GAME_VALUES: Mapping[MacroName, MacroValue] = {
    MacroName("OIT_WAVELET_RANK"): MacroValue("2"),
    MacroName("OIT_COEFF_COUNT"): MacroValue("8"),  # LevelRenderer.OIT_COEFFICIENT_COUNT, 2 to the power of 3
    MacroName("OIT_COEFF_ATTACHMENT_COUNT"): MacroValue("2"),  # a quarter of the coefficient count
    MacroName("PORTAL_LAYERS"): MacroValue("16"),  # 15 for the end portal and 16 for the end gateway
}

# Tested names that the game switches on with a number rather than with nothing, since the shaders compare with
# them, as in `color.a < ALPHA_CUTOUT`. The game gives 0.1 to most pipelines and 0.5 to a few; either compiles alike.
_SWITCH_VALUES: Mapping[MacroName, MacroValue] = {MacroName("ALPHA_CUTOUT"): MacroValue("0.1")}

# The game switches a name on with an empty value.
_SWITCHED_ON = MacroValue("")

_CORE = "core/"
_INCLUDE = "include/"
_INCLUDE_LINE = re.compile(r"^\s*#\s*include\s*<minecraft:(?P<name>[^>]+)>\s*$")
_DIRECTIVE = re.compile(r"^\s*#\s*(?P<word>\w+)\s*(?P<rest>.*?)\s*$")
_DEFINITION = re.compile(r"(?P<name>[A-Za-z_]\w*)(?P<value>.*)$")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_TESTED = re.compile(r"\bdefined\s*(?:\(\s*(?P<within>[A-Za-z_]\w*)\s*\)|(?P<bare>[A-Za-z_]\w*))")
_COMMENT_START = re.compile(r"//|/\*")
_LINE_COMMENT = "//"
_BLOCK_END = "*/"
_CONTINUED = "\\"  # a line that ends with this goes on on the next line
_WORD_GAP = " "  # what a comment inside a line leaves, so that the words either side stay apart
_LINE_BREAK = "\n"


class CompilerNotFound(Exception):
    """The reference GLSL compiler is not installed, so no shader can be checked."""


class IncludeNotFound(Exception):
    """A shader includes a file that neither the pack nor the game has."""

    def __init__(self, include: ShaderPath, by: ShaderPath):
        super().__init__(f"{by} includes {include}, which neither the pack nor the game has")
        self.include = include
        self.by = by


class IncludeCycle(Exception):
    """A shader includes a file that, through its own includes, includes the shader again."""

    def __init__(self, include: ShaderPath, by: ShaderPath):
        super().__init__(f"{by} includes {include}, which includes {by} again")
        self.include = include
        self.by = by


@dataclass(frozen=True)
class CompileFailure:
    """A shader that does not compile: which one, the defines it was compiled under, and what the compiler said."""

    shader: ShaderPath
    defines: Defines
    message: CompilerMessage


class _Stage(Enum):
    """The step of drawing a core shader runs in, as the end of its file name tells it."""

    VERTEX = ".vsh"
    FRAGMENT = ".fsh"


_STAGE_NAMES: Mapping[_Stage, Argument] = {_Stage.VERTEX: Argument("vert"), _Stage.FRAGMENT: Argument("frag")}


@dataclass(frozen=True)
class _Line:
    """One line of a shader with its includes written out, and the file and line it came from."""

    text: GlslLine
    path: ShaderPath
    number: LineNumber


@dataclass(frozen=True)
class _Unit:
    """One run of the compiler: the text it reads, and the command that runs it."""

    command: tuple[Argument, ...]
    text: ShaderText


@dataclass(frozen=True)
class _Check:
    """One shader under one combination of defines, and the vanilla original it is held against, if it has one."""

    defines: Defines
    built: _Unit
    original: _Unit | None


@dataclass(frozen=True)
class _Target:
    """A shader to compile: its lines, which say where each message points, and each combination to compile it in."""

    path: ShaderPath
    lines: tuple[_Line, ...]
    checks: tuple[_Check, ...]


@dataclass(frozen=True)
class _Run:
    """What the compiler did with one unit."""

    unit: _Unit
    exit_code: ExitCode
    output: CompilerOutput


# ---- which shaders, and their text ------------------------------------------------------------------------------

@deal.pure
def _stage(path: ShaderPath) -> _Stage | None:
    """The stage of a core shader; None for any other file."""
    if not path.startswith(_CORE):
        return None
    return next((s for s in _Stage if path.endswith(s.value)), None)


@deal.has()
@deal.raises(IncludeNotFound, IncludeCycle)
def _resolve(path: ShaderPath, files: Mapping[ShaderPath, ShaderText],
             within: tuple[ShaderPath, ...] = ()) -> tuple[_Line, ...]:
    """The lines of a shader with each `#include <minecraft:name>` replaced by the lines of include/name.

    The game resolves includes the same way, as text, and leaves repeats to the guards inside the files.
    """
    lines: list[_Line] = []
    for number, text in enumerate(files[path].splitlines(), start=1):
        match = _INCLUDE_LINE.match(text)
        if match is None:
            lines.append(_Line(GlslLine(text), path, LineNumber(number)))
            continue
        include = ShaderPath(_INCLUDE + match["name"])
        if include == path or include in within:
            raise IncludeCycle(include, path)
        if include not in files:
            raise IncludeNotFound(include, path)
        lines.extend(_resolve(include, files, (*within, path)))
    return tuple(lines)


@deal.pure
def _text(lines: tuple[_Line, ...]) -> ShaderText:
    """The text the compiler reads."""
    return ShaderText(_LINE_BREAK.join(line.text for line in lines) + _LINE_BREAK)


# ---- the preprocessor, as far as telling variants apart needs it ------------------------------------------------

class _Kind(Enum):
    """What a step of a shader is to the preprocessor."""

    IF = "if"
    IFDEF = "ifdef"
    IFNDEF = "ifndef"
    ELIF = "elif"
    ELSE = "else"
    ENDIF = "endif"
    DEFINE = "define"
    UNDEF = "undef"
    TEXT = "text"  # lines the preprocessor keeps or drops, including directives that do not choose


_OPENS = frozenset({_Kind.IF, _Kind.IFDEF, _Kind.IFNDEF})
_CHOOSES = _OPENS | {_Kind.ELIF, _Kind.ELSE, _Kind.ENDIF}
_KINDS: Mapping[str, _Kind] = {k.value: k for k in _Kind if k is not _Kind.TEXT}


class _Operator(Enum):
    """An operator of a `#if` expression."""

    OR = "||"
    AND = "&&"
    EQUAL = "=="
    UNEQUAL = "!="
    AT_MOST = "<="
    AT_LEAST = ">="
    LESS = "<"
    GREATER = ">"
    PLUS = "+"
    MINUS = "-"
    TIMES = "*"
    DIVIDED = "/"
    REMAINDER = "%"
    NOT = "!"
    NEGATIVE = "unary -"
    POSITIVE = "unary +"
    OPEN = "("
    CLOSE = ")"


# How tightly each operator binds, as in C.
_BINDING: Mapping[_Operator, int] = {
    _Operator.OR: 1, _Operator.AND: 2, _Operator.EQUAL: 3, _Operator.UNEQUAL: 3, _Operator.AT_MOST: 4,
    _Operator.AT_LEAST: 4, _Operator.LESS: 4, _Operator.GREATER: 4, _Operator.PLUS: 5, _Operator.MINUS: 5,
    _Operator.TIMES: 6, _Operator.DIVIDED: 6, _Operator.REMAINDER: 6, _Operator.NOT: 7, _Operator.NEGATIVE: 7,
    _Operator.POSITIVE: 7,
}
_PREFIXES: Mapping[_Operator, _Operator] = {
    _Operator.NOT: _Operator.NOT, _Operator.MINUS: _Operator.NEGATIVE, _Operator.PLUS: _Operator.POSITIVE}
_UNARY: Mapping[_Operator, Callable[[int], int | bool]] = {
    _Operator.NOT: operator.not_, _Operator.NEGATIVE: operator.neg, _Operator.POSITIVE: operator.pos}
_BINARY: Mapping[_Operator, Callable[[int, int], int | bool]] = {
    _Operator.OR: operator.or_, _Operator.AND: operator.and_, _Operator.EQUAL: operator.eq,
    _Operator.UNEQUAL: operator.ne, _Operator.AT_MOST: operator.le, _Operator.AT_LEAST: operator.ge,
    _Operator.LESS: operator.lt, _Operator.GREATER: operator.gt, _Operator.PLUS: operator.add,
    _Operator.MINUS: operator.sub, _Operator.TIMES: operator.mul,
}
_LOGICAL = frozenset({_Operator.OR, _Operator.AND})  # these take their operands as true or false
_DIVISIONS = frozenset({_Operator.DIVIDED, _Operator.REMAINDER})
_DEFINED_WORD = "defined"
_EXPRESSION_TOKEN = re.compile(r"\s*(?:(?P<number>\d+)[uU]?(?![\w.])|(?P<name>[A-Za-z_]\w*)|(?P<operator>"
                               + "|".join(re.escape(o.value) for o in _Operator if len(o.value) <= 2) + r"))")


class _TermKind(Enum):
    """What a term of an expression in postfix order is."""

    NUMBER = "number"
    NAME = "name"
    DEFINED = "defined"
    OPERATOR = "operator"


@dataclass(frozen=True)
class _Term:
    """One term of a `#if` expression, in postfix order."""

    kind: _TermKind
    number: int = 0
    name: MacroName = MacroName("")
    operator: _Operator = _Operator.OR


@dataclass(frozen=True)
class _Step:
    """A directive of a shader, or a run of lines between two directives that choose or define."""

    kind: _Kind
    name: MacroName = MacroName("")  # what #ifdef, #ifndef, #define and #undef name
    value: MacroValue = MacroValue("")  # what #define gives the name
    condition: tuple[_Term, ...] | None = None  # the expression of #if and #elif; None when it cannot be read
    words: frozenset[str] = frozenset()  # the identifiers of the lines a run or a #define holds


@dataclass(frozen=True)
class _Key:
    """What the preprocessor makes of a shader under one combination: the steps it keeps and the switched-on names
    that those steps use. Two combinations with the same key give the compiler the same text."""

    kept: tuple[int, ...]
    used: frozenset[MacroName]


@deal.pure
def _code(lines: tuple[_Line, ...]) -> tuple[str, ...]:
    """Each line without its comments; a line wholly inside a block comment is left empty."""
    code: list[str] = []
    commented = False
    for line in lines:
        rest, kept = str(line.text), ""
        while rest:
            if commented:
                end = rest.find(_BLOCK_END)
                commented = end < 0
                rest = "" if commented else rest[end + len(_BLOCK_END):]
                continue
            start = _COMMENT_START.search(rest)
            if start is None:
                kept, rest = kept + rest, ""
            elif start.group() == _LINE_COMMENT:
                kept, rest = kept + rest[:start.start()], ""
            else:
                kept, rest, commented = kept + rest[:start.start()] + _WORD_GAP, rest[start.end():], True
        code.append(kept)
    return tuple(code)


@deal.pure
def _guards(files: Mapping[ShaderPath, ShaderText]) -> frozenset[MacroName]:
    """The include guards of the include files: the name of an `#ifndef` that opens a file and is followed by a
    `#define`.

    A guard is told by its shape rather than by the name it defines, since 26.3's terrainglobals.glsl tests
    MINECRAFT_TERRAINGLOBLAS_GLSL and defines MINECRAFT_TERRAINGLOBALS_GLSL.
    """
    guards: set[MacroName] = set()
    for path, text in files.items():
        if not path.startswith(_INCLUDE):
            continue
        lines = tuple(_Line(GlslLine(t), path, LineNumber(n)) for n, t in enumerate(text.splitlines(), start=1))
        directives = [m for m in map(_DIRECTIVE.match, _code(lines)) if m is not None][:2]
        named = _DEFINITION.match(directives[0]["rest"]) if directives else None
        if named is not None and [d["word"] for d in directives] == [_Kind.IFNDEF.value, _Kind.DEFINE.value]:
            guards.add(MacroName(named["name"]))
    return frozenset(guards)


@deal.pure
def _tested(code: tuple[str, ...], guards: frozenset[MacroName]) -> Defines:
    """The names a shader tests, in alphabetical order: those of #ifdef, #ifndef and `defined`, leaving out include
    guards and the values the game always gives."""
    names: set[MacroName] = set()
    for text in code:
        match = _DIRECTIVE.match(text)
        if match is None:
            continue
        word, rest = match["word"], match["rest"]
        named = _DEFINITION.match(rest)
        if word in (_Kind.IFDEF.value, _Kind.IFNDEF.value) and named is not None:
            names.add(MacroName(named["name"]))
        elif word in (_Kind.IF.value, _Kind.ELIF.value):
            names.update(MacroName(m["within"] or m["bare"]) for m in _TESTED.finditer(rest))
    return tuple(sorted(names - guards - set(_GAME_VALUES)))


@deal.pure
def _condition(expression: str) -> tuple[_Term, ...] | None:
    """A `#if` expression in postfix order; None when it is not one this module can work out."""
    tokens: list[re.Match[str]] = []
    at = 0
    while at < len(expression.rstrip()):
        token = _EXPRESSION_TOKEN.match(expression, at)
        if token is None:
            return None
        tokens.append(token)
        at = token.end()
    terms: list[_Term] = []
    pending: list[_Operator] = []
    wants_operand = True
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if token["number"] is not None and wants_operand:
            terms.append(_Term(_TermKind.NUMBER, number=int(token["number"])))
            wants_operand = False
        elif token["name"] == _DEFINED_WORD and wants_operand:
            words = [t.group().strip() for t in tokens[index:index + 3]]
            if words[:1] and _IDENTIFIER.fullmatch(words[0]):
                terms.append(_Term(_TermKind.DEFINED, name=MacroName(words[0])))
                index += 1
            elif len(words) == 3 and _IDENTIFIER.fullmatch(words[1]) and words[2] == _Operator.CLOSE.value:
                terms.append(_Term(_TermKind.DEFINED, name=MacroName(words[1])))
                index += 3
            else:
                return None
            wants_operand = False
        elif token["name"] is not None and wants_operand:
            terms.append(_Term(_TermKind.NAME, name=MacroName(token["name"])))
            wants_operand = False
        elif token["operator"] is None:
            return None
        else:
            operator = _Operator(token["operator"])
            if operator is _Operator.OPEN and wants_operand:
                pending.append(operator)
            elif operator is _Operator.CLOSE and not wants_operand:
                while pending and pending[-1] is not _Operator.OPEN:
                    terms.append(_Term(_TermKind.OPERATOR, operator=pending.pop()))
                if not pending:
                    return None
                pending.pop()
            elif wants_operand and operator in _PREFIXES:
                pending.append(_PREFIXES[operator])
            elif not wants_operand and operator in _BINDING:
                while (pending and pending[-1] is not _Operator.OPEN
                       and _BINDING[pending[-1]] >= _BINDING[operator]):
                    terms.append(_Term(_TermKind.OPERATOR, operator=pending.pop()))
                pending.append(operator)
                wants_operand = True
            else:
                return None
    if wants_operand or _Operator.OPEN in pending:
        return None
    terms.extend(_Term(_TermKind.OPERATOR, operator=o) for o in reversed(pending))
    return tuple(terms)


@deal.pure
def _steps(code: tuple[str, ...]) -> tuple[_Step, ...] | None:
    """A shader as the preprocessor walks it; None when it holds a directive this module cannot follow."""
    steps: list[_Step] = []
    run: set[str] = set()
    for text in code:
        match = _DIRECTIVE.match(text)
        kind = _KINDS.get(match["word"], _Kind.TEXT) if match is not None else _Kind.TEXT
        if kind is not _Kind.TEXT and text.rstrip().endswith(_CONTINUED):
            return None
        if kind is _Kind.TEXT:
            run.update(_IDENTIFIER.findall(text))
            continue
        if run:
            steps.append(_Step(_Kind.TEXT, words=frozenset(run)))
            run = set()
        rest = match["rest"] if match is not None else ""
        if kind in (_Kind.IF, _Kind.ELIF):
            steps.append(_Step(kind, condition=_condition(rest)))
        elif kind in (_Kind.ELSE, _Kind.ENDIF):
            steps.append(_Step(kind))
        else:
            definition = _DEFINITION.match(rest)
            if definition is None:
                return None
            steps.append(_Step(kind, name=MacroName(definition["name"]),
                               value=MacroValue(definition["value"].strip()),
                               words=frozenset(_IDENTIFIER.findall(rest))))
    if run:
        steps.append(_Step(_Kind.TEXT, words=frozenset(run)))
    return tuple(steps)


@deal.pure
def _macros(defines: Defines) -> dict[MacroName, MacroValue]:
    """Each name the compiler is given, with its value, when the given names are switched on."""
    return {**_GAME_VALUES, **{name: _SWITCH_VALUES.get(name, _SWITCHED_ON) for name in defines}}


@deal.pure
def _evaluate(condition: tuple[_Term, ...] | None, macros: Mapping[MacroName, MacroValue]) -> int | None:
    """The value of a `#if` expression; None when it cannot be worked out, as when a name stands for no number."""
    if condition is None:
        return None
    stack: list[int] = []
    for term in condition:
        if term.kind is _TermKind.NUMBER:
            stack.append(term.number)
        elif term.kind is _TermKind.DEFINED:
            stack.append(int(term.name in macros))
        elif term.kind is _TermKind.NAME:
            value = macros.get(term.name)
            if value is not None and not value.isdigit():
                return None
            stack.append(int(value) if value is not None else 0)  # C counts an undefined name as 0
        elif term.operator in _UNARY and stack:
            stack.append(int(_UNARY[term.operator](stack.pop())))
        elif len(stack) < 2:
            return None
        else:
            right, left = stack.pop(), stack.pop()
            if term.operator in _DIVISIONS:
                if right == 0:
                    return None
                quotient = abs(left) // abs(right) * (1 if (left < 0) == (right < 0) else -1)  # C rounds to 0
                stack.append(quotient if term.operator is _Operator.DIVIDED else left - right * quotient)
            elif term.operator in _LOGICAL:
                stack.append(int(_BINARY[term.operator](bool(left), bool(right))))
            else:
                stack.append(int(_BINARY[term.operator](left, right)))
    return stack[0] if len(stack) == 1 else None


@deal.pure
def _key(steps: tuple[_Step, ...], defines: Defines) -> _Key | None:
    """What the preprocessor makes of a shader when the given names are switched on; None when that cannot be told."""
    macros = _macros(defines)
    enclosing: list[tuple[bool, bool]] = []  # for each open #if: whether its group is live, and whether a branch was
    live = True
    kept: list[int] = []
    for index, step in enumerate(steps):
        if step.kind in _CHOOSES and step.kind not in _OPENS and not enclosing:
            return None
        if step.kind in _OPENS:
            holds = (_evaluate(step.condition, macros) if step.kind is _Kind.IF
                     else int((step.name in macros) == (step.kind is _Kind.IFDEF))) if live else 0
            if holds is None:
                return None
            enclosing.append((live, bool(holds)))
            live = bool(holds)
        elif step.kind is _Kind.ELIF:
            outer, taken = enclosing[-1]
            holds = _evaluate(step.condition, macros) if outer and not taken else 0
            if holds is None:
                return None
            live = bool(holds)
            enclosing[-1] = (outer, taken or live)
        elif step.kind is _Kind.ELSE:
            outer, taken = enclosing[-1]
            live = outer and not taken
            enclosing[-1] = (outer, True)
        elif step.kind is _Kind.ENDIF:
            live = enclosing.pop()[0]
        elif live:
            kept.append(index)
            if step.kind is _Kind.DEFINE:
                macros[step.name] = step.value
            elif step.kind is _Kind.UNDEF:
                macros.pop(step.name, None)
    if enclosing:
        return None
    used = frozenset(name for name in defines if any(name in steps[i].words for i in kept))
    return _Key(tuple(kept), used)


@deal.pure
def _combinations(names: Defines) -> tuple[Defines, ...]:
    """Every combination of the names, fewest first, so that a failure is named with as few defines as it needs."""
    return tuple(itertools.chain.from_iterable(itertools.combinations(names, n) for n in range(len(names) + 1)))


@deal.pure
def _variants(code: tuple[str, ...], names: Defines) -> tuple[Defines, ...]:
    """One combination of each set of combinations that give the compiler the same text.

    Compiling the same text twice can only give the same verdict, and this is what keeps the number of compiler
    runs within the time a build has. When the text cannot be told apart, every combination is its own variant.
    """
    combinations = _combinations(names)
    steps = _steps(code)
    if steps is None:
        return combinations
    variants: dict[_Key, Defines] = {}
    for defines in combinations:
        key = _key(steps, defines)
        if key is None:
            return combinations
        variants.setdefault(key, defines)
    return tuple(variants.values())


# ---- the plan, the command, and the verdict ---------------------------------------------------------------------

@deal.pure
def _unit(stage: _Stage, text: ShaderText, defines: Defines) -> _Unit:
    """The compiler run for a text under a combination of defines, with the values the game always gives."""
    definitions = (Argument(f"{_DEFINE}{name}={value}") for name, value in sorted(_macros(defines).items()))
    return _Unit((COMPILER, *_RULES, *_NO_BINARY, *_FROM_INPUT, _STAGE_NAMES[stage], *definitions), text)


@deal.has()
@deal.raises(IncludeNotFound, IncludeCycle)
@deal.ensure(lambda built, vanilla, result: all(_stage(t.path) is not None for t in result))
@deal.ensure(lambda built, vanilla, result: {f.path for f in built if _stage(f.path) is not None}
             <= {t.path for t in result})
@deal.ensure(lambda built, vanilla, result: all(t.checks and t.checks[0].defines == () for t in result))
@deal.ensure(lambda built, vanilla, result: all((c.original is not None) == (t.path in {f.path for f in vanilla})
                                                for t in result for c in t.checks))
def _plan(built: tuple[ShaderFile, ...], vanilla: tuple[ShaderFile, ...]) -> tuple[_Target, ...]:
    """Each shader to compile, with the combinations to compile it in.

    These are the built core shaders, and the vanilla core shaders that reach an include that the pack changes or
    adds, since such an include changes them too.
    """
    originals = {f.path: f.text for f in vanilla}
    files = {**originals, **{f.path: f.text for f in built}}
    changed = frozenset(f.path for f in built if f.path.startswith(_INCLUDE) and originals.get(f.path) != f.text)
    rebuilt = frozenset(f.path for f in built)
    guards = _guards(files)
    stages = {path: stage for path in sorted(files) if (stage := _stage(path)) is not None}
    targets: list[_Target] = []
    for path, stage in stages.items():
        lines = _resolve(path, files)
        if path not in rebuilt and not changed & {line.path for line in lines}:
            continue
        text = _text(lines)
        original = _text(_resolve(path, originals)) if path in originals else None
        code = _code(lines)
        checks = tuple(_Check(d, _unit(stage, text, d), _unit(stage, original, d) if original is not None else None)
                       for d in _variants(code, _tested(code, guards)))
        targets.append(_Target(path, lines, checks))
    return tuple(targets)


@deal.pure
@deal.ensure(lambda targets, result: len(set(result)) == len(result))
@deal.ensure(lambda targets, result: set(result) == {c.built for t in targets for c in t.checks})
def _built_units(targets: tuple[_Target, ...]) -> tuple[_Unit, ...]:
    """Each compiler run the built shaders need, once."""
    return tuple(dict.fromkeys(c.built for t in targets for c in t.checks))


@deal.pure
@deal.pre(lambda targets, runs: {c.built for t in targets for c in t.checks} <= {r.unit for r in runs})
@deal.ensure(lambda targets, runs, result: len(set(result)) == len(result))
@deal.ensure(lambda targets, runs, result: _originals_of_failures(targets, runs) == set(result))
def _original_units(targets: tuple[_Target, ...], runs: tuple[_Run, ...]) -> tuple[_Unit, ...]:
    """Each compiler run of a vanilla original that a failure needs, to tell whether the game uses that combination,
    leaving out what has been run already."""
    done = {r.unit: r.exit_code for r in runs}
    wanted = (c.original for t in targets for c in t.checks
              if c.original is not None and done[c.built] != _SUCCESS)
    return tuple(u for u in dict.fromkeys(wanted) if u not in done)


@deal.pure
def _place(number: LineNumber, lines: tuple[_Line, ...]) -> str:
    """Where a line of the text the compiler read came from, as `path:line:`."""
    if not 0 < number <= len(lines):
        return f"{lines[-1].path if lines else _INPUT_NAME}:{_END}:"
    line = lines[number - 1]
    return f"{line.path}:{line.number}:"


@deal.pure
def _message(output: CompilerOutput, lines: tuple[_Line, ...]) -> CompilerMessage:
    """What the compiler said, with each place in the text it read given as the file and line it came from."""
    said: list[str] = []
    for text in output.splitlines():
        if not text.strip() or text == _INPUT_NAME:
            continue
        parts, at = [], 0
        for match in _INPUT_PLACE.finditer(text):
            parts += [text[at:match.start()], _place(LineNumber(int(match["line"])), lines)]
            at = match.end()
        said.append("".join([*parts, text[at:]]))
    return CompilerMessage(_LINE_BREAK.join(said))


@deal.pure
def _originals_of_failures(targets: tuple[_Target, ...], runs: tuple[_Run, ...]) -> set[_Unit]:
    """The vanilla originals of the combinations whose built shader failed, that were not run already."""
    exit_codes = {r.unit: r.exit_code for r in runs}
    failed = (c for t in targets for c in t.checks if exit_codes[c.built] != _SUCCESS)
    return {c.original for c in failed if c.original is not None} - set(exit_codes)


@deal.pure
def _reports_only_failures(targets: tuple[_Target, ...], runs: tuple[tuple[_Run, ...], ...],
                           failures: tuple[CompileFailure, ...]) -> bool:
    """Whether each failure is a combination whose built shader failed, and is excused unless its shader's
    original failed under every combination."""
    exit_codes = {r.unit: r.exit_code for group in runs for r in group}
    checks = {(t.path, c.defines): c for t in targets for c in t.checks}
    hopeless = {t.path for t in targets
                if all(c.original is not None and exit_codes.get(c.original) not in (None, _SUCCESS) for c in t.checks)}
    for failure in failures:
        check = checks.get((failure.shader, failure.defines))
        if check is None or exit_codes[check.built] == _SUCCESS:
            return False
        original_failed = check.original is not None and exit_codes[check.original] != _SUCCESS
        if original_failed and failure.shader not in hopeless:
            return False
    return True


@deal.pure
def _all_run(targets: tuple[_Target, ...], runs: tuple[tuple[_Run, ...], ...]) -> bool:
    """Whether each built unit was run, and the original of each built unit that failed."""
    exit_codes = {r.unit: r.exit_code for group in runs for r in group}
    checks = [c for t in targets for c in t.checks]
    return all(c.built in exit_codes for c in checks) and all(
        c.original in exit_codes for c in checks if c.original is not None and exit_codes[c.built] != _SUCCESS)


@deal.pure
@deal.pre(lambda targets, *runs: _all_run(targets, runs))
@deal.ensure(lambda targets, *runs, result: len({(f.shader, f.message) for f in result}) == len(result))
@deal.ensure(lambda targets, *runs, result: _reports_only_failures(targets, runs, result))
def _failures(targets: tuple[_Target, ...], *runs: tuple[_Run, ...]) -> tuple[CompileFailure, ...]:
    """Each way a shader fails to compile, once, under the fewest defines it fails under.

    A combination under which the vanilla original fails too is not one the game uses. But when the original fails
    under every combination, the compiler is not being asked the way the game asks it, so nothing is excused.
    """
    done = {r.unit: r for group in runs for r in group}
    failures: dict[tuple[ShaderPath, CompilerMessage], CompileFailure] = {}
    for target in targets:
        failed = [c for c in target.checks if done[c.built].exit_code != _SUCCESS]
        original_fails = [c.original is not None and done[c.original].exit_code != _SUCCESS for c in failed]
        forgiving = len(failed) < len(target.checks) or not all(original_fails)
        for check, original_failed in zip(failed, original_fails):
            if original_failed and forgiving:
                continue
            message = _message(done[check.built].output, target.lines)
            failures.setdefault((target.path, message), CompileFailure(target.path, check.defines, message))
    return tuple(failures.values())


# ---- running the compiler ---------------------------------------------------------------------------------------

@deal.raises(CompilerNotFound, IncludeNotFound, IncludeCycle)
def validate(built: tuple[ShaderFile, ...], vanilla: tuple[ShaderFile, ...]) -> tuple[CompileFailure, ...]:
    """Each way a built shader fails to compile; empty when the pack is safe to write.

    `built` is what the pack changes or adds: core shaders and include files. `vanilla` is every shader file of the
    game. A built include wins over the vanilla one of the same path.

    Raises CompilerNotFound when glslangValidator is not installed, and IncludeNotFound or IncludeCycle when an
    include cannot be written out.
    """
    targets = _plan(built, vanilla)
    first = _compile(_built_units(targets))
    second = _compile(_original_units(targets, first))
    return _failures(targets, first, second)


def _compile(units: tuple[_Unit, ...]) -> tuple[_Run, ...]:
    # One compiler process starts in about 25 ms and a shader can have hundreds of variants, so they run side by
    # side; the threads only wait on processes.
    with ThreadPoolExecutor() as pool:
        return tuple(pool.map(_run, units))


def _run(unit: _Unit) -> _Run:
    try:
        done = subprocess.run(unit.command, input=unit.text, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, check=False)
    except FileNotFoundError as missing:
        raise CompilerNotFound(f"{COMPILER} is not installed") from missing
    return _Run(unit, ExitCode(done.returncode), CompilerOutput(done.stdout))
