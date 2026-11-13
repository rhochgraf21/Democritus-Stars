"""Gives the textures of water, leaves and plants an alpha value that tells the shaders what the surface is.

The game's terrain shader is told a position, a colour, a texture coordinate and a light level, and
nothing about which block a surface belongs to. The block texture is the one thing the pack controls that
reaches the shader, so its alpha carries the mark.

An effect recognises a mark by reading the alpha of one texel of level 0 of the block atlas, with
texelFetch or textureLod at the texel's centre, and comparing it with alpha_of(surface) / 255 to within
half of 1 / 255. Two reasons for level 0 at a centre: the game's own sampleNearest blends neighbouring
texels near a texel's edge, and the smaller mipmap levels average a sprite's texels. Either can make a
leaf's mark and its transparent holes average out to a value that is no mark, or make glass and its holes
average out to one that is.
"""

from dataclasses import dataclass, replace
from enum import Enum

import deal

from shader.vanilla import Channel, Pixel, Texture, TexturePath


class Surface(Enum):
    """A kind of surface that the shaders tell apart by the mark on its texture."""

    WATER = "water"
    LEAVES = "leaves"
    PLANT = "plant"


# The alpha of a texel that the game discards, so that holes in leaves and plants stay holes.
TRANSPARENT = Channel(0)

# The alpha of every texel of an opaque vanilla texture: so common that it can never be a mark.
OPAQUE = Channel(255)

# Vanilla water is 180 in every texel, and its alpha is how see-through it is, so its mark has to stay
# near 180. 180 itself is also slime's, and the nether portal, drawn translucent like water, holds every
# value from 155 to 232. 154 is the nearest value to 180 that no other block texture of 26.3 holds. The
# water effect, which recognises the mark, can give water back its vanilla 180 if the look asks for it.
_WATER_MARK = Channel(154)

# Leaves and plants are cut out: a texel is kept or discarded, so any alpha well above the game's
# threshold looks the same as 255. No block texture of 26.3 holds a value from 239 to 254. The marks sit
# two steps from 255 and from each other, which leaves a whole step between a mark and its neighbour at a
# tolerance of half a step.
#
# Incomplete: 26.3 chooses the layer of each quad from its texture. A texture with any alpha other than 0
# and 255 is drawn in the translucent layer, not the cutout one (NativeImage.computeTransparency,
# ChunkSectionLayer.byTransparency). So these two marks move leaves and plants into the translucent layer,
# which sorts and blends them. Whether that costs too much of the frame rate is to be tried in the game.
_LEAVES_MARK = Channel(253)
_PLANT_MARK = Channel(251)

_MARKS = {
    Surface.WATER: _WATER_MARK,
    Surface.LEAVES: _LEAVES_MARK,
    Surface.PLANT: _PLANT_MARK,
}

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


@dataclass
class UnmarkedTexture(Exception):
    """A texture that is not of water, leaves or a plant, so the markers have no mark for it."""

    path: TexturePath

    def __str__(self) -> str:
        return f"{self.path} is not a texture of water, leaves or a plant"


@deal.pure
@deal.post(lambda result: TRANSPARENT < result < OPAQUE)
def alpha_of(surface: Surface) -> Channel:
    """The alpha that marks a texel of this kind of surface, for an effect to compare in its GLSL."""
    return _MARKS[surface]


@deal.pure
@deal.ensure(lambda path, result: (result is None) == (path not in MARKED))
@deal.ensure(lambda path, result: result is None or path in _TEXTURES_OF[result])
def surface_of(path: TexturePath) -> Surface | None:
    """The kind of surface that a block texture is, or None if it is none that the shaders tell apart."""
    return _SURFACE_OF.get(path)


def _is_marked_from(texture: Texture, marked: Texture) -> bool:
    surface = surface_of(texture.path)
    if surface is None:
        return False
    mark = alpha_of(surface)
    return (
        (marked.path, marked.width, marked.height, marked.meta)
        == (texture.path, texture.width, texture.height, texture.meta)
        and len(marked.pixels) == len(texture.pixels)
        and all(_is_pixel_marked_from(pixel, after, mark) for pixel, after in zip(texture.pixels, marked.pixels))
    )


def _is_pixel_marked_from(pixel: Pixel, marked: Pixel, mark: Channel) -> bool:
    expected = TRANSPARENT if pixel.alpha == TRANSPARENT else mark
    return (marked.red, marked.green, marked.blue, marked.alpha) == (pixel.red, pixel.green, pixel.blue, expected)


def _marked_pixel(pixel: Pixel, mark: Channel) -> Pixel:
    # A transparent texel keeps 0: any other alpha fills in the holes of leaves and plants.
    if pixel.alpha == TRANSPARENT:
        return pixel
    return replace(pixel, alpha=mark)


@deal.has()
@deal.raises(UnmarkedTexture)
def _marked(texture: Texture) -> Texture:
    surface = surface_of(texture.path)
    if surface is None:
        raise UnmarkedTexture(texture.path)
    mark = alpha_of(surface)
    return replace(texture, pixels=tuple(_marked_pixel(pixel, mark) for pixel in texture.pixels))


@deal.has()
@deal.raises(UnmarkedTexture)
@deal.ensure(lambda textures, result: len(result) == len(textures))
@deal.ensure(lambda textures, result: all(_is_marked_from(*pair) for pair in zip(textures, result)))
def mark(textures: tuple[Texture, ...]) -> tuple[Texture, ...]:
    """The textures with their marks: each texel that is not transparent takes the mark of its surface.

    Path, size, animation and colours are kept. Raises UnmarkedTexture for a texture that is not one of
    MARKED, rather than pass it on unmarked, since a caller who hands one over has asked for the wrong one.
    """
    return tuple(_marked(texture) for texture in textures)
