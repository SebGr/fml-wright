import logging
import geopandas as gpd
import pandas as pd

from fmlwright.dataset_generators import ImageBaseGenerator
from fmlwright.core import preprocessing, labeling

log = logging.getLogger(__name__)


class ImageSingleStepGenerator(ImageBaseGenerator):
    """Class to create complete model steps from geodata."""

    def find_overlap(self, gdf1, gdf2):
        """Find overlap between gdf.

        Args:
            gdf1: geodataframe of which you want to have the overlapping elements returned
            gdf2: geodataframe that the overlap needs to be found against.
        """
        gdf1.crs = "epsg:4326"  # todo: fix this ugliness
        gdf2.crs = "epsg:4326"
        return gpd.sjoin(gdf1, gdf2).drop("index_right", axis=1)

    def _create_buffer_size(self, areas_gdf):
        """Create the buffer size."""
        AREA_SIZE_DIVIDER = 30000
        MIN_BUFFER_SIZE = 3
        MAX_BUFFER_SIZE = 10
        wall_gdf = areas_gdf.loc[areas_gdf["category"] == "wall"].copy()

        area = wall_gdf.unary_union.buffer(0.01).convex_hull.area

        buffer_size = area / AREA_SIZE_DIVIDER
        buffer_size = max(buffer_size, MIN_BUFFER_SIZE)
        buffer_size = min(buffer_size, MAX_BUFFER_SIZE)
        return buffer_size

    def create_input_gdf(self, areas_gdf):
        """Create input gdf.

        Args:
            areas_gdf (gpd.GeoDataFrame):  Geodataframe of single house.

        Returns:
            Geodataframe prepared to be turned into an image.
        """
        relevant_floorplan_gdf = areas_gdf.copy()
        buffer_size = self._create_buffer_size(relevant_floorplan_gdf)

        extended_doors = relevant_floorplan_gdf.loc[
            relevant_floorplan_gdf["category"] == "door"
        ].copy()
        extended_doors.geometry = extended_doors.buffer(buffer_size * 2, cap_style=2)
        extended_doors["colors"] = "blue"

        outer_walls = preprocessing.create_outer_bounderies(
            relevant_floorplan_gdf.loc[
                relevant_floorplan_gdf["category"] != "balcony"
            ].copy(),
            buffer=0.1,
        )

        entrance = relevant_floorplan_gdf.loc[
            relevant_floorplan_gdf["category"] == "entrance"
        ].copy()

        windows = self.find_overlap(extended_doors, outer_walls)
        windows["colors"] = "red"

        CATEGORIES_TO_DROP = ["wall", "door", "entrance"]
        areas_gdf = areas_gdf.loc[
            ~areas_gdf["category"].isin(CATEGORIES_TO_DROP)
        ].copy()

        floors_gdf = gpd.GeoDataFrame(geometry=[areas_gdf.unary_union])
        floors_gdf["colors"] = [[0.0, 0.0, 0.0]]
        floors_gdf["category"] = ["floor"]
        floors_gdf = floors_gdf.explode().reset_index(drop=True)

        if not entrance.empty:
            entrance_gdf = self.find_overlap(windows, entrance)
            entrance_gdf["colors"] = "green"
            return pd.concat([floors_gdf, windows, entrance_gdf])
        return pd.concat([floors_gdf, windows])

    def create_output_gdf(self, areas_gdf):
        """Create the output image for a single step model.

        Args:
            areas_gdf: geodataframe of a house.

        Returns:
            Geodataframe prepared to be turned into an image.
        """
        buffer_size = self._create_buffer_size(areas_gdf)
        wall_gdf = areas_gdf.loc[areas_gdf["category"] == "wall"].copy()

        CATEGORIES_TO_DROP = ["door", "entrance", "wall"]
        rooms_gdf = areas_gdf.loc[
            ~areas_gdf["category"].isin(CATEGORIES_TO_DROP)
        ].copy()

        extended_doors = areas_gdf.loc[areas_gdf["category"] == "door"].copy()
        extended_doors.geometry = extended_doors.buffer(buffer_size * 2, cap_style=2)

        walls_final = gpd.overlay(wall_gdf, extended_doors, how="difference")
        walls_final.geometry = walls_final.buffer(
            buffer_size, join_style=2, cap_style=3
        )
        walls_final["colors"] = [[0.0, 0.0, 0.0]] * walls_final.shape[0]

        rooms_gdf = rooms_gdf.sort_values("area_size", ascending=False)
        colormap = labeling.get_room_color_map()
        colors = pd.DataFrame(
            list(
                rooms_gdf["category"]
                .apply(lambda cat: [x / 255 for x in colormap[cat]])
                .values
            ),
            columns=["color_r", "color_g", "color_b"],
        )
        rooms_gdf["colors"] = list(colors[["color_r", "color_g", "color_b"]].values)

        return pd.concat([rooms_gdf, walls_final])
