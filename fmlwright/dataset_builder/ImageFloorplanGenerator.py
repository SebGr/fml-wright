import logging

import pandas as pd

from fmlwright.core import labeling
from fmlwright.dataset_builder import ImageBaseGenerator

log = logging.getLogger(__name__)


class ImageFloorplanGenerator(ImageBaseGenerator):
    """Class to create floorplan images from geodata."""

    def create_input_gdf(self, areas_gdf):
        """Create the input image. This is a black image as the groundplan.

        Args:
            areas_gdf (gpd.GeoDataFrame): Geodataframe of single house

        Returns:
            Geodataframe prepared to be turned into an image.
        """
        areas_gdf = areas_gdf.copy()
        areas_gdf["colors"] = "black"
        return areas_gdf

    def create_output_gdf(self, areas_gdf):
        """Create the output image. This is an geodataframe with colors for the separate rooms.

        Args:
            areas_gdf (gpd.GeoDataFrame): Geodataframe of single house

        Returns:
            Geodataframe prepared to be turned into an image.
        """
        areas_gdf = areas_gdf.copy()
        CATEGORIES_TO_DROP = ["wall", "door", "entrance"]
        areas_gdf = areas_gdf.loc[
            ~areas_gdf["category"].isin(CATEGORIES_TO_DROP)
        ].copy()
        colormap = labeling.get_room_color_map()
        colors = pd.DataFrame(
            list(
                areas_gdf["category"]
                .apply(lambda cat: [x / 255 for x in colormap[cat]])
                .values
            ),
            columns=["color_r", "color_g", "color_b"],
        )
        areas_gdf["colors"] = list(colors[["color_r", "color_g", "color_b"]].values)

        areas_gdf = areas_gdf.sort_values("area_size", ascending=False)
        return areas_gdf
