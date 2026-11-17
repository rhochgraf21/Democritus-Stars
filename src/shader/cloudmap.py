"""The game's cloud texture with every cell filled.

The game reads its clouds from one texture: each pixel is a cell of the cloud layer, 12 blocks wide, and
a cell is empty when its pixel's alpha is under 10 (CloudRenderer.isCellEmpty of 26.3). The cloud shaders
are sent a face only for a cell that is not empty. The clouds effect draws its own clouds anywhere in the
layer, so every cell has to send a face, which a texture with no empty pixel does.

This and the clouds effect belong together: with the map filled and the game's own cloud shaders, the sky
is one unbroken slab of cloud. A build that leaves the effect out has to leave the map out.
"""

from dataclasses import replace

import deal

from shader.vanilla import Channel, Pixel, Texture, TexturePath

# Where the game keeps its cloud texture.
CLOUD_MAP = TexturePath("environment/clouds.png")

# Full in every channel. The alpha is what fills the cell; white is the colour the game's own cells have.
_FULL = Channel(255)
FILLED_CELL = Pixel(_FULL, _FULL, _FULL, _FULL)


class MissingCloudMap(Exception):
    """The cloud texture was not among the textures given."""


@deal.has()
@deal.raises(MissingCloudMap)
@deal.ensure(lambda vanilla, result: result.path == CLOUD_MAP and all(pixel == FILLED_CELL for pixel in result.pixels))
@deal.ensure(lambda vanilla, result: any(
    (before.path, before.width, before.height, before.meta, len(before.pixels))
    == (result.path, result.width, result.height, result.meta, len(result.pixels))
    for before in vanilla
))
def filled(vanilla: tuple[Texture, ...]) -> Texture:
    """The game's cloud texture, its size kept, with every cell filled.

    Textures that are not the cloud texture are passed over. MissingCloudMap if it is not given.
    """
    given = [texture for texture in vanilla if texture.path == CLOUD_MAP]
    if not given:
        raise MissingCloudMap(f"the game's textures lack the cloud texture {CLOUD_MAP}")
    return replace(given[0], pixels=tuple(FILLED_CELL for _ in given[0].pixels))
