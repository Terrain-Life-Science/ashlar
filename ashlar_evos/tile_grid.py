"""
Tile grid generation for tiled image processing.

Generates overlapping tile grids for efficient memory-efficient processing
of large images.
"""

from typing import List, NamedTuple, Tuple

import numpy as np


class TileInfo(NamedTuple):
    """Information about a single tile."""

    y: int  # Starting row position
    x: int  # Starting column position
    height: int  # Tile height
    width: int  # Tile width
    tile_idx: int  # Tile index in grid


class TileGrid:
    """Generate and manage overlapping tile grids for large images."""

    def __init__(self, image_shape: Tuple[int, int], tile_size: int = 4096, overlap: int = 512):
        """
        Initialize tile grid.

        Parameters
        ----------
        image_shape : tuple
            (height, width) of the full image
        tile_size : int
            Size of each tile (tiles are square)
        overlap : int
            Overlap between adjacent tiles in pixels
        """
        self.image_shape = image_shape
        self.tile_size = tile_size
        self.overlap = overlap
        self.tiles = self._generate_tiles()

    def _generate_tiles(self) -> List[TileInfo]:
        """
        Generate list of tiles covering the image.

        Returns
        -------
        list
            List of TileInfo objects
        """
        h, w = self.image_shape
        tiles = []
        tile_idx = 0

        # Calculate step size (tile_size - overlap)
        step = self.tile_size - self.overlap

        # Generate tiles row by row
        y = 0
        while y < h:
            # Adjust height for last row
            tile_height = min(self.tile_size, h - y)

            x = 0
            while x < w:
                # Adjust width for last column
                tile_width = min(self.tile_size, w - x)

                tile = TileInfo(y=y, x=x, height=tile_height, width=tile_width, tile_idx=tile_idx)
                tiles.append(tile)
                tile_idx += 1

                # Move to next column
                x += step
                if x >= w:
                    break

            # Move to next row
            y += step
            if y >= h:
                break

        return tiles

    def __len__(self) -> int:
        """Get number of tiles."""
        return len(self.tiles)

    def __iter__(self):
        """Iterate over tiles."""
        return iter(self.tiles)

    def __getitem__(self, idx: int) -> TileInfo:
        """Get tile by index."""
        return self.tiles[idx]

    def get_tile_at_position(self, y: int, x: int) -> TileInfo:
        """
        Get tile that contains the given position.

        Parameters
        ----------
        y : int
            Row position
        x : int
            Column position

        Returns
        -------
        TileInfo
            Tile containing the position
        """
        for tile in self.tiles:
            if tile.y <= y < tile.y + tile.height and tile.x <= x < tile.x + tile.width:
                return tile
        raise ValueError(f"Position ({y}, {x}) not found in any tile")

    def get_grid_dimensions(self) -> Tuple[int, int]:
        """
        Get grid dimensions (number of rows, number of columns).

        Returns
        -------
        tuple
            (num_rows, num_cols)
        """
        if len(self.tiles) == 0:
            return (0, 0)

        # Find max y and x positions
        max_y = max(tile.y + tile.height for tile in self.tiles)
        max_x = max(tile.x + tile.width for tile in self.tiles)

        # Calculate grid dimensions
        step = self.tile_size - self.overlap
        num_rows = int(np.ceil((max_y - self.overlap) / step))
        num_cols = int(np.ceil((max_x - self.overlap) / step))

        return (num_rows, num_cols)

    def get_tile_coordinates(self, tile_idx: int) -> Tuple[int, int]:
        """
        Get grid coordinates (row, col) for a tile.

        Parameters
        ----------
        tile_idx : int
            Tile index

        Returns
        -------
        tuple
            (row, col) in the grid
        """
        tile = self.tiles[tile_idx]
        step = self.tile_size - self.overlap
        row = tile.y // step
        col = tile.x // step
        return (row, col)
