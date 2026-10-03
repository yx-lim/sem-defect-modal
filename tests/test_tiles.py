import numpy as np

from sem.tiles import TILE, STRIDE, extract_tiles, pad_image, tile_coords


def test_tiles_cover_image():
    h, w = 2316, 6996
    coords = tile_coords(h, w)
    padded, ph, pw = pad_image(np.zeros((h, w), np.uint8))
    # every tile inside padded image
    for y, x in coords:
        assert 0 <= y <= ph - TILE and 0 <= x <= pw - TILE
    # coverage: union of tiles covers padded image
    cover = np.zeros((ph, pw), bool)
    for y, x in coords:
        cover[y:y + TILE, x:x + TILE] = True
    assert cover.all()
    assert ph >= h and pw >= w


def test_extract_tiles_shape():
    img = np.random.default_rng(0).integers(0, 255, (1000, 1500), dtype=np.uint8)
    tiles, coords, phw = extract_tiles(img)
    assert tiles.shape[1:] == (TILE, TILE)
    assert len(tiles) == len(coords)
