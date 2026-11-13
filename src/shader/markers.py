"""Gives the textures of water, leaves and plants a mark in their texels that tells the shaders what the surface is.

The game's terrain shader is told a position, a colour, a texture coordinate and a light level, and
nothing about which block a surface belongs to. The block texture is the one thing the pack controls that
reaches the shader, so its texels carry the mark.

Water carries an AlphaMark: every texel that is not transparent has that alpha. Leaves and plants carry a
Signature in the low bits of the colour of each opaque texel; their alpha is left alone, because 26.3
draws any quad whose texture holds an alpha other than 0 and 255 in the translucent layer, not the cutout
one (NativeImage.computeTransparency, ChunkSectionLayer.byTransparency).

An effect reads a mark with texelFetch at mipmap level 0. The game's own sampleNearest blends neighbouring
texels near a texel's edge, and the smaller levels average a sprite's texels, so either could lose a mark
or make one up. One texel of level 0 is the texel as the pack wrote it, with one exception: the game
rewrites the colour of transparent texels when it builds the atlas (TextureUtil.solidify and
fillEmptyAreasWithDarkColor, called from MipmapGenerator.generateMipLevels). That is why only opaque texels
carry a signature, and why its reader looks at opaque texels only.

A vertex of leaves or a plant knows only the corner of its sprite. The wind effect steps from that corner
to the sprite's centre and looks at PROBES there in order: the first opaque one decides, and the surface is
leaves or a plant if that texel's low colour bits are the surface's signature. A wrong guess of which way
the sprite lies lands in the middle of a neighbouring sprite, where no texture outside MARKED carries one.

Not yet seen in the game, and settled the first time the pack is: that the atlas puts no padding between
sprites, and that a quad's vertex order tells which way its sprite lies from each corner.
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import NewType

import deal

from shader.vanilla import Channel, Pixel, PixelCount, Texture, TexturePath


class Surface(Enum):
    """A kind of surface that the shaders tell apart by the mark on its texture."""

    WATER = "water"
    LEAVES = "leaves"
    PLANT = "plant"


# The lowest and the highest value of a colour channel.
CHANNEL_MIN = Channel(0)
CHANNEL_MAX = Channel(255)

# The alpha of a texel that the game discards, so that holes in leaves and plants stay holes.
TRANSPARENT = CHANNEL_MIN

# The alpha of a wholly opaque texel: the only texels whose colour the game leaves as the pack wrote it.
OPAQUE = CHANNEL_MAX

# How many of the lowest bits of a colour channel a signature takes.
BitCount = NewType("BitCount", int)

# The value of the lowest SIGNATURE_BITS bits of one colour channel.
LowBits = NewType("LowBits", int)

# Where a probe sits from the centre texel of a sprite, in texels: positive is to the right or down.
TexelOffset = NewType("TexelOffset", int)


@dataclass(frozen=True)
class AlphaMark:
    """A mark in alpha: every texel of the surface that is not transparent has this alpha."""

    alpha: Channel


@dataclass(frozen=True)
class Signature:
    """A mark in colour: every opaque texel of the surface has these lowest bits in red, green and blue."""

    red: LowBits
    green: LowBits
    blue: LowBits


@dataclass(frozen=True)
class Probe:
    """A texel that the reader of a signature looks at, by where it sits from the centre texel of a sprite.

    The centre texel is the one in column SPRITE_SIDE / 2 and row SPRITE_SIDE / 2 of its frame, counted
    from 0 at the top left: the texel whose top left corner is the centre of the sprite.
    """

    across: TexelOffset
    down: TexelOffset


# Three bits of each channel. With two, every one of the 64 patterns is already held by some opaque texel
# in the middle of a block texture of 26.3, so no signature could keep stone still. With three, 131 of the
# 512 patterns are held by none, and rounding to the nearest value with a signature's bits moves a channel
# by at most 4 of 255 (7 at the ends of the range), which the eye does not see.
SIGNATURE_BITS = BitCount(3)

_SIGNATURE_STEP = 1 << SIGNATURE_BITS

_NO_BITS = LowBits(0)
_ALL_BITS = LowBits(_SIGNATURE_STEP - 1)

# All low bits zero is where flat and hand-drawn colours such as 0, 128 or 248 fall, and all ones is where
# 255 falls: a signature on either would be found on unmarked textures first.
_COMMON_PATTERNS = ((_NO_BITS, _NO_BITS, _NO_BITS), (_ALL_BITS, _ALL_BITS, _ALL_BITS))

# Vanilla water is 180 in every texel, and its alpha is how see-through it is, so its mark has to stay
# near 180. 180 itself is also slime's, and the nether portal, drawn translucent like water, holds every
# value from 155 to 232. 154 is the nearest value to 180 that no other block texture of 26.3 holds. The
# water effect, which recognises the mark, can give water back its vanilla 180 if the look asks for it.
_WATER_MARK = AlphaMark(Channel(154))

# Of the patterns that the first opaque probe of no unmarked block texture of 26.3 holds, these two have
# the fewest held patterns one step away in a single channel, which leaves the most room for a texture a
# later version adds. They differ in every channel, and neither is all zeros nor all ones, the patterns
# that flat colours fall on most.
_LEAVES_SIGNATURE = Signature(LowBits(4), LowBits(6), LowBits(4))
_PLANT_SIGNATURE = Signature(LowBits(6), LowBits(2), LowBits(1))

_MARKS: dict[Surface, AlphaMark | Signature] = {
    Surface.WATER: _WATER_MARK,
    Surface.LEAVES: _LEAVES_SIGNATURE,
    Surface.PLANT: _PLANT_SIGNATURE,
}

# The side of one frame of a vanilla block sprite; an animated one stacks its frames from the top. The
# wind effect steps half of it from a corner to reach the centre.
SPRITE_SIDE = PixelCount(16)

_CENTRE = SPRITE_SIDE // 2

# Five are the fewest probes that find an opaque texel in every frame of every leaf and plant texture of
# 26.3, and each is a texture read per vertex. They were found by a search over those textures, so they
# follow no pattern: the first is opaque in most textures, and the later ones catch the thin stems of young
# crops, the sparse glow of firefly bushes and eyeblossoms, and the bent frames of seagrass.
PROBES: tuple[Probe, ...] = (
    Probe(TexelOffset(-1), TexelOffset(6)),
    Probe(TexelOffset(-3), TexelOffset(1)),
    Probe(TexelOffset(0), TexelOffset(-5)),
    Probe(TexelOffset(-1), TexelOffset(-3)),
    Probe(TexelOffset(3), TexelOffset(5)),
)

_BLOCK_FOLDER = "block/"
_IMAGE_SUFFIX = ".png"


def _block_textures(*names: str) -> tuple[TexturePath, ...]:
    return tuple(TexturePath(_BLOCK_FOLDER + name + _IMAGE_SUFFIX) for name in names)


# The names are listed one by one, from the game's own list of block textures, rather than matched by a
# pattern such as "*_leaves": a pattern that catches a new texture by its name can catch the wrong one,
# and a wrong mark makes stone sway or glass ripple. A name the game drops is noticed, because the game
# no longer has the texture asked for; a texture the game adds is unmarked until it is listed here.

_WATER = _block_textures("water_still", "water_flow", "water_overlay")

# Bamboo's leaves are left out: they sit on a stalk that does not sway, and would come away from it.
_LEAVES = _block_textures(
    "acacia_leaves",
    "azalea_leaves",
    "birch_leaves",
    "cherry_leaves",
    "dark_oak_leaves",
    "flowering_azalea_leaves",
    "jungle_leaves",
    "mangrove_leaves",
    "oak_leaves",
    "orange_poplar_leaves",
    "pale_oak_leaves",
    "red_poplar_leaves",
    "spruce_leaves",
    "yellow_poplar_leaves",
)

# Plants that stand up from the ground they grow on, so their foot can stay put while the rest sways.
# Left out: what hangs (vines, roots, hanging moss, cave vines, the hanging propagule) or sits on a wall
# (lichen, shelf mushrooms, cocoa), which would need a fixed top or side; what lies flat (lily pads,
# petals, wildflowers, leaf litter, moss carpet); what is stiff (corals, mushrooms, dripleaves, amethyst,
# bamboo, cactus, chorus); and stems attached to a melon or pumpkin, which would come away from the fruit.
# A potted flower uses the flower's own texture, and sways in its pot with it.

_GRASSES = _block_textures(
    "short_grass",
    "tall_grass_bottom",
    "tall_grass_top",
    "short_dry_grass",
    "tall_dry_grass",
    "fern",
    "large_fern_bottom",
    "large_fern_top",
    "bush",
    "dead_bush",
    "red_shrub",
    # The glow is drawn over the bush with the same shape, and has to sway with it.
    "firefly_bush",
    "firefly_bush_emissive",
)

_FLOWERS = _block_textures(
    "allium",
    "azure_bluet",
    "blue_orchid",
    "cactus_flower",
    "closed_eyeblossom",
    "cornflower",
    "dandelion",
    "golden_dandelion",
    "lily_of_the_valley",
    # The glow is drawn over the flower with the same shape, and has to sway with it.
    "open_eyeblossom",
    "open_eyeblossom_emissive",
    "orange_tulip",
    "oxeye_daisy",
    "pink_tulip",
    "poppy",
    "red_tulip",
    "torchflower",
    "white_tulip",
    "wither_rose",
    "lilac_bottom",
    "lilac_top",
    "peony_bottom",
    "peony_top",
    "rose_bush_bottom",
    "rose_bush_top",
    # The head is part of the top half, and has to sway with its stem.
    "sunflower_bottom",
    "sunflower_top",
    "sunflower_front",
    "sunflower_back",
)

_SAPLINGS = _block_textures(
    "acacia_sapling",
    "birch_sapling",
    "cherry_sapling",
    "dark_oak_sapling",
    "jungle_sapling",
    "oak_sapling",
    "pale_oak_sapling",
    "poplar_sapling",
    "spruce_sapling",
    "mangrove_propagule",
)

# The pitcher crop's pod is a box, and stays still; only its leaves are listed.
_CROPS = _block_textures(
    "wheat_stage0",
    "wheat_stage1",
    "wheat_stage2",
    "wheat_stage3",
    "wheat_stage4",
    "wheat_stage5",
    "wheat_stage6",
    "wheat_stage7",
    "carrots_stage0",
    "carrots_stage1",
    "carrots_stage2",
    "carrots_stage3",
    "potatoes_stage0",
    "potatoes_stage1",
    "potatoes_stage2",
    "potatoes_stage3",
    "beetroots_stage0",
    "beetroots_stage1",
    "beetroots_stage2",
    "beetroots_stage3",
    "torchflower_crop_stage0",
    "torchflower_crop_stage1",
    "pitcher_crop_bottom_stage_1",
    "pitcher_crop_bottom_stage_2",
    "pitcher_crop_bottom_stage_3",
    "pitcher_crop_bottom_stage_4",
    "pitcher_crop_top_stage_3",
    "pitcher_crop_top_stage_4",
    "sweet_berry_bush_stage0",
    "sweet_berry_bush_stage1",
    "sweet_berry_bush_stage2",
    "sweet_berry_bush_stage3",
    "melon_stem",
    "pumpkin_stem",
    "sugar_cane",
)

# Under water the current moves them as the wind moves grass.
_WATER_PLANTS = _block_textures(
    "seagrass",
    "tall_seagrass_bottom",
    "tall_seagrass_top",
    "kelp",
    "kelp_plant",
)

_NETHER_PLANTS = _block_textures(
    "crimson_roots",
    "warped_roots",
    "nether_sprouts",
    "crimson_fungus",
    "warped_fungus",
    "nether_wart_stage0",
    "nether_wart_stage1",
    "nether_wart_stage2",
)

_TEXTURES_OF = {
    Surface.WATER: _WATER,
    Surface.LEAVES: _LEAVES,
    Surface.PLANT: _GRASSES + _FLOWERS + _SAPLINGS + _CROPS + _WATER_PLANTS + _NETHER_PLANTS,
}

_SURFACE_OF = {path: surface for surface, paths in _TEXTURES_OF.items() for path in paths}

# The block textures that the markers mark: the ones to ask the game for.
#
# A list of paths, and not a function that picks the marked ones out of all the game's textures: when a
# new version renames or drops one, asking for it fails and the change is noticed, where a picker would
# quietly pick nothing for it.
MARKED: tuple[TexturePath, ...] = tuple(_SURFACE_OF)

# Which frame of an animated texture, counted from the top from 0.
FrameIndex = NewType("FrameIndex", int)


class MarkError(Exception):
    """A texture that the markers cannot mark."""


@dataclass
class UnmarkedTexture(MarkError):
    """A texture that is not of water, leaves or a plant, so the markers have no mark for it."""

    path: TexturePath

    def __str__(self) -> str:
        return f"{self.path} is not a texture of water, leaves or a plant"


@dataclass
class UnexpectedSpriteSize(MarkError):
    """A leaf or plant texture whose frames are not SPRITE_SIDE square, so the wind effect misses its centre."""

    path: TexturePath
    width: PixelCount
    height: PixelCount

    def __str__(self) -> str:
        return f"{self.path} is {self.width} by {self.height}, not frames of {SPRITE_SIDE} by {SPRITE_SIDE}"


@dataclass
class UnreadableFrame(MarkError):
    """A leaf or plant texture with a frame in which no probe is opaque, so its signature could never be read."""

    path: TexturePath
    frame: FrameIndex

    def __str__(self) -> str:
        return f"{self.path}: frame {self.frame} has no opaque texel at any probe, so it would never sway"


def _is_sound(mark: AlphaMark | Signature) -> bool:
    if isinstance(mark, AlphaMark):
        return TRANSPARENT < mark.alpha < OPAQUE
    bits = (mark.red, mark.green, mark.blue)
    return all(_NO_BITS <= bit <= _ALL_BITS for bit in bits) and bits not in _COMMON_PATTERNS


@deal.pure
@deal.post(_is_sound)
@deal.ensure(lambda surface, result: isinstance(result, AlphaMark) == (surface is Surface.WATER))
def mark_of(surface: Surface) -> AlphaMark | Signature:
    """The mark of a kind of surface, for an effect to compare in its GLSL: water's alpha, or a signature."""
    return _MARKS[surface]


@deal.pure
@deal.ensure(lambda path, result: (result is None) == (path not in MARKED))
@deal.ensure(lambda path, result: result is None or path in _TEXTURES_OF[result])
def surface_of(path: TexturePath) -> Surface | None:
    """The kind of surface that a block texture is, or None if it is none that the shaders tell apart."""
    return _SURFACE_OF.get(path)


def _low_bits(channel: Channel) -> LowBits:
    return LowBits(channel % _SIGNATURE_STEP)


def _carries(pixel: Pixel, signature: Signature) -> bool:
    return (_low_bits(pixel.red), _low_bits(pixel.green), _low_bits(pixel.blue)) == (
        signature.red,
        signature.green,
        signature.blue,
    )


def _with_low_bits(channel: Channel, bits: LowBits) -> Channel:
    # The nearest value with these bits, so that no channel moves by more than half a step unless the
    # nearest one lies outside the range.
    same_step = channel - _low_bits(channel) + bits
    candidates = (same_step - _SIGNATURE_STEP, same_step, same_step + _SIGNATURE_STEP)
    return Channel(min((c for c in candidates if CHANNEL_MIN <= c <= CHANNEL_MAX), key=lambda c: abs(c - channel)))


def _signed(pixel: Pixel, signature: Signature) -> Pixel:
    # The game rewrites the colour of transparent texels, and a partly transparent one is never read.
    if pixel.alpha != OPAQUE:
        return pixel
    return replace(
        pixel,
        red=_with_low_bits(pixel.red, signature.red),
        green=_with_low_bits(pixel.green, signature.green),
        blue=_with_low_bits(pixel.blue, signature.blue),
    )


def _alpha_marked(pixel: Pixel, mark: AlphaMark) -> Pixel:
    # A transparent texel keeps 0: the game discards it, and so it stays a hole.
    if pixel.alpha == TRANSPARENT:
        return pixel
    return replace(pixel, alpha=mark.alpha)


def _frames(texture: Texture) -> tuple[FrameIndex, ...]:
    return tuple(FrameIndex(frame) for frame in range(texture.height // SPRITE_SIDE))


def _probe_texel(texture: Texture, frame: FrameIndex, probe: Probe) -> Pixel:
    column = _CENTRE + probe.across
    row = frame * SPRITE_SIDE + _CENTRE + probe.down
    return texture.pixels[row * texture.width + column]


def _is_readable(texture: Texture, frame: FrameIndex) -> bool:
    return any(_probe_texel(texture, frame, probe).alpha == OPAQUE for probe in PROBES)


@deal.has()
@deal.raises(UnexpectedSpriteSize, UnreadableFrame)
def _check_readable(texture: Texture) -> None:
    if texture.width != SPRITE_SIDE or texture.height % SPRITE_SIDE != 0 or texture.height == 0:
        raise UnexpectedSpriteSize(texture.path, texture.width, texture.height)
    for frame in _frames(texture):
        if not _is_readable(texture, frame):
            raise UnreadableFrame(texture.path, frame)


@deal.has()
@deal.raises(UnmarkedTexture, UnexpectedSpriteSize, UnreadableFrame)
def _marked(texture: Texture) -> Texture:
    surface = surface_of(texture.path)
    if surface is None:
        raise UnmarkedTexture(texture.path)
    mark = mark_of(surface)
    if isinstance(mark, AlphaMark):
        return replace(texture, pixels=tuple(_alpha_marked(pixel, mark) for pixel in texture.pixels))
    _check_readable(texture)
    return replace(texture, pixels=tuple(_signed(pixel, mark) for pixel in texture.pixels))


def _is_alpha_marked_from(pixel: Pixel, marked: Pixel, mark: AlphaMark) -> bool:
    alpha = TRANSPARENT if pixel.alpha == TRANSPARENT else mark.alpha
    return marked == replace(pixel, alpha=alpha)


def _is_signed_from(pixel: Pixel, marked: Pixel, signature: Signature) -> bool:
    if pixel.alpha != OPAQUE:
        return marked == pixel
    moves = (marked.red - pixel.red, marked.green - pixel.green, marked.blue - pixel.blue)
    return (
        marked.alpha == pixel.alpha
        and _carries(marked, signature)
        and all(abs(move) < _SIGNATURE_STEP for move in moves)
    )


def _is_marked_from(texture: Texture, marked: Texture) -> bool:
    surface = surface_of(texture.path)
    if surface is None or len(marked.pixels) != len(texture.pixels):
        return False
    if (marked.path, marked.width, marked.height, marked.meta) != (
        texture.path,
        texture.width,
        texture.height,
        texture.meta,
    ):
        return False
    mark = mark_of(surface)
    pairs = zip(texture.pixels, marked.pixels)
    if isinstance(mark, AlphaMark):
        return all(_is_alpha_marked_from(pixel, after, mark) for pixel, after in pairs)
    return all(_is_signed_from(pixel, after, mark) for pixel, after in pairs)


@deal.has()
@deal.raises(UnmarkedTexture, UnexpectedSpriteSize, UnreadableFrame)
@deal.ensure(lambda textures, result: len(result) == len(textures))
@deal.ensure(lambda textures, result: all(_is_marked_from(*pair) for pair in zip(textures, result)))
def mark(textures: tuple[Texture, ...]) -> tuple[Texture, ...]:
    """The textures with their marks, each texture's path, size, animation and alpha otherwise kept.

    Water: each texel that is not transparent takes the water's alpha, and its colour is kept. Leaves and
    plants: each opaque texel takes the surface's signature, moving each channel to the nearest value with
    those low bits, and every other texel is left as it is.

    Raises UnmarkedTexture for a texture that is not one of MARKED, rather than pass it on unmarked, since
    a caller who hands one over has asked for the wrong one. Raises UnexpectedSpriteSize or UnreadableFrame
    for a leaf or plant texture whose signature the wind effect could not read: it would never sway.
    """
    return tuple(_marked(texture) for texture in textures)
