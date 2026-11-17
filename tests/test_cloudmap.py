"""The cloud map fills every cell of the game's cloud texture, and nothing else about it changes."""

import pytest
from zigdag import verifies

from shader.cloudmap import CLOUD_MAP, FILLED_CELL, MissingCloudMap, filled
from shader.vanilla import Channel, Pixel, PixelCount, Texture, TextureMeta, TexturePath

# The alpha under which the game takes a cell for empty (CloudRenderer.isCellEmpty of 26.3).
EMPTY_BELOW = Channel(10)

EMPTY = Pixel(Channel(0), Channel(0), Channel(0), Channel(0))
CLOUD = Pixel(Channel(255), Channel(255), Channel(255), Channel(255))
FAINT = Pixel(Channel(255), Channel(255), Channel(255), Channel(9))

# A cloud texture three cells wide and two high, with cells the game would leave empty.
GAME_MAP = Texture(CLOUD_MAP, PixelCount(3), PixelCount(2), (EMPTY, CLOUD, FAINT, CLOUD, EMPTY, EMPTY), None)
OTHER = Texture(TexturePath("block/stone.png"), PixelCount(1), PixelCount(1), (EMPTY,), TextureMeta("{}"))


@verifies("shader.cloudmap#fill")
def test_no_cell_of_the_filled_map_is_one_the_game_takes_for_empty() -> None:
    assert all(pixel.alpha >= EMPTY_BELOW for pixel in filled((GAME_MAP,)).pixels)
    assert FILLED_CELL.alpha >= EMPTY_BELOW


@verifies("shader.cloudmap#fill")
def test_the_filled_map_keeps_the_place_and_the_size_of_the_game_s() -> None:
    made = filled((OTHER, GAME_MAP))
    assert (made.path, made.width, made.height, made.meta) == (CLOUD_MAP, GAME_MAP.width, GAME_MAP.height, None)
    assert len(made.pixels) == len(GAME_MAP.pixels)


@verifies("shader.cloudmap#fill")
def test_without_the_game_s_cloud_texture_there_is_nothing_to_fill() -> None:
    with pytest.raises(MissingCloudMap):
        filled((OTHER,))
