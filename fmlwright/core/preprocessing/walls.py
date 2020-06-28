import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon

from fmlwright.core.labeling import get_room_color_map
from fmlwright.core.utils import find_intersecting_points, cut_line_at_points


def create_subsection_walls(walls_gdf):
    """This function creates wall subsections for intersecting pieces of walls.

    Args:
        walls_gdf (gpd.GeoDataFrame): walls geodataframe

    Returns:
        Pandas dataframe containing walls and subsections.
    """
    walls = []
    for idx, row in walls_gdf.iterrows():
        intersecting_walls = walls_gdf.loc[walls_gdf.index != idx].copy()
        main_wall = walls_gdf.loc[[idx]]

        points = find_intersecting_points(intersecting_walls, main_wall)

        cut_main_wall_sub_parts = cut_line_at_points(
            main_wall.geometry.iloc[0], [x for x in points.geometry]
        )
        walls += cut_main_wall_sub_parts

    # remove supersets
    walls = gpd.GeoDataFrame(geometry=walls).drop_duplicates()
    walls = walls.loc[walls.geometry.length != 0].copy()
    walls = walls.reset_index(drop=False)

    walls = pd.concat(
        [
            walls,
            (
                pd.DataFrame(
                    walls.geometry.apply(lambda x: list(x.coords)).tolist(),
                    columns=["start_point", "end_point"],
                )
            ),
        ],
        axis=1,
    )
    return walls


def generate_outer_walls(areas_gdf, buffer_size=2.5):
    """Generate the outer walls.

    Args:
        areas_gdf:
        buffer_size:

    Returns:
        Dataframe containing the outer walls.
    """
    colormap = get_room_color_map()
    walls = gpd.GeoDataFrame(
        geometry=areas_gdf.exterior.buffer(buffer_size, join_style=2)
    )
    walls["category"] = "wall"
    walls["colors"] = (
        walls["category"].apply(lambda cat: [x / 255 for x in colormap[cat]]).values
    )
    return walls


def create_walls_image(walls, thickness=5):
    """Create an image of the walls.

    Args:
        walls:
        thickness:

    Returns:
        cv2 image with the walls.
    """
    min_x, min_y, max_x, max_y = [int(x) for x in walls.total_bounds]
    image = np.zeros((max_y, max_x), dtype=np.uint8).copy()

    for i, row in walls.iterrows():
        cv2.line(image, row.start, row.end, thickness=thickness, color=255)
    return image


def create_outer_bounderies(walls, buffer=5):
    """Create the outer boundaries.

    Args:
        walls:
        buffer:

    Returns:
        Geodataframe containing outer boundaries.
    """
    result_polygon = walls.unary_union.buffer(buffer, join_style=2)
    if isinstance(result_polygon, MultiPolygon):
        outer_boundaries = gpd.GeoDataFrame(geometry=list(result_polygon))
    else:
        outer_boundaries = gpd.GeoDataFrame(geometry=[result_polygon])
    outer_boundaries.geometry = outer_boundaries.exterior
    return outer_boundaries
