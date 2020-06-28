import geopandas as gpd
import numpy as np
from scipy.spatial import Voronoi
from shapely import ops
from shapely.geometry import Point, LineString, Polygon

from fmlwright.core.utils import create_exterior_points, get_angle_between_points
from fmlwright.core.preprocessing import (
    image_to_geodataframe,
    resize_and_pad,
    clear_overlap_floorplan,
)

import warnings

# Geopandas update about changes in notna.
# Otherwise gets called during function gpd.clip in create_voronoi_gdf.
warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)


def create_voronoi_gdf(coordinates, input_gdf):
    """Create a voronoi gdf given a list of coordinates and an input image.

    The exterior of the floorplan image will be used as an outer limit for the geodataframe.

    Args:
        coordinates (list): List of tuples containing points.
        input_gdf (gpd.GeoDataFrame): original b&w input image gdf.

    Returns:
        Geodataframe with voronoi areas, clipped to the input image.
    """
    BUFFER_SIZE = 20
    edge_coordinates = create_exterior_points(
        input_gdf.buffer(BUFFER_SIZE, join_style=2), number_points_along_line=10
    )

    vor = Voronoi(coordinates + edge_coordinates)

    lines = [
        LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line
    ]
    all_polygons = list(ops.polygonize(lines))

    areas = gpd.GeoDataFrame(geometry=all_polygons)

    clipped_gdf = gpd.clip(areas, input_gdf, keep_geom_type=True)
    return clipped_gdf


def straighten_polygon(geom_shape):
    """Straighten a polygon in angles of 90 degrees.

    Args:
        geom_shape (Polygon): Polygon to straighten

    Returns:
        Straightened polygon.
    """
    edges_coords = list(geom_shape.exterior.coords)

    points = [Point(edges_coords[0])]
    cur_point = edges_coords[0]
    next_point = cur_point
    for i in np.arange(len(edges_coords) - 1):
        angle = get_angle_between_points(cur_point, edges_coords[i + 1])
        all_angles = [0, 90, 180, 270, 360]
        nearest_angle = min(all_angles, key=lambda x: abs(x - angle))

        if nearest_angle in [0, 180, 360]:
            # keep x val, take y
            next_point = (edges_coords[i + 1][0], cur_point[1])

        if nearest_angle in [90, 270]:
            next_point = (cur_point[0], edges_coords[i + 1][1])

        cur_point = next_point
        points.append(Point(next_point))
    return Polygon(points)


def create_voronoi_floorplan(floorplan_gdf, input_gdf):
    """Create and process a floorplan geodataframe using voronoi algorithm.

    Args:
        floorplan_gdf (gpd.GeoDataFrame): Floorplan geodataframe containing polygons.
        input_gdf (gpd.GeoDataFrame): original b&w input image gdf.

    Returns:
        Floorplan geodataframe clipped to the input_gdf with voronoi algorithm applied to it.
    """
    locs = create_exterior_points(floorplan_gdf, number_points_along_line=10)
    voronoi_gdf = create_voronoi_gdf(locs, input_gdf)
    voronoi_gdf = gpd.sjoin(voronoi_gdf, floorplan_gdf, how="left")
    voronoi_gdf = voronoi_gdf.dissolve(by="index_right").reset_index(drop=True)

    voronoi_gdf = voronoi_gdf.append(floorplan_gdf)
    voronoi_gdf = voronoi_gdf.dissolve(by="category").reset_index().explode()

    isna_index = voronoi_gdf["colors"].isna()
    voronoi_gdf.loc[isna_index, "category"] = "unknown"
    voronoi_gdf.loc[isna_index, "colors"] = voronoi_gdf.loc[isna_index, "colors"].apply(
        lambda x: [x / 255 for x in [255, 230, 180]]
    )
    clipped_gpf = gpd.clip(voronoi_gdf, input_gdf)
    clipped_gpf = (
        clipped_gpf.dissolve(by="category")
        .reset_index()
        .explode()
        .reset_index(drop=True)
    )
    return clipped_gpf


def create_straightened_floorplan(floorplan_gdf, input_gdf, buffer_size=1):
    """Straighten a gdf to multiples of 90 degree angles.

    Args:
        floorplan_gdf (gpd.GeoDataFrame): Floorplan geodataframe containing polygons.
        input_gdf (gpd.GeoDataFrame): original b&w input image gdf.
        buffer_size (int): Size to buffer the straightened polygons with.

    Returns:
        Floorplan geodataframe clipped to the input_gdf with straightened polygons.
    """
    fixed_polygons = [straighten_polygon(x) for x in floorplan_gdf.geometry.values]
    straightened_gdf = floorplan_gdf.copy()
    straightened_gdf.geometry = fixed_polygons
    straightened_gdf.geometry = straightened_gdf.buffer(buffer_size, join_style=2)
    straightened_gdf = straightened_gdf.loc[~straightened_gdf.is_empty].copy()
    clipped_gpf = gpd.clip(straightened_gdf, input_gdf)
    return clipped_gpf


def process_floorplan(floorplan_gdf, input_gdf, tolerance=0.5):
    """Postprocess a floorplan.

    This first does voronoi and then straightening on the floorplan_geodataframe. Finally it
    simplifies the final gdf with the set tolerance.

    Args:
        floorplan_gdf (gpd.GeoDataFrame): Floorplan geodataframe containing polygons.
        input_gdf (gpd.GeoDataFrame): original b&w input image gdf.
        tolerance:

    Returns:
        Geodataframe with straightened polygons.
    """
    voronoi_gdf = create_voronoi_floorplan(floorplan_gdf, input_gdf)
    straightened_gdf = create_straightened_floorplan(
        voronoi_gdf, input_gdf, buffer_size=3
    )
    straightened_gdf.geometry = straightened_gdf.simplify(tolerance=tolerance)
    straightened_gdf = straightened_gdf.dissolve(by="category").reset_index().explode()

    straightened_gdf = create_straightened_floorplan(
        straightened_gdf, input_gdf, buffer_size=3
    )
    straightened_gdf.geometry = straightened_gdf.simplify(tolerance=tolerance)
    straightened_gdf = straightened_gdf.dissolve(by="category").reset_index().explode()

    straightened_gdf["area"] = straightened_gdf.area
    straightened_gdf = (
        straightened_gdf.loc[straightened_gdf.area != 0].reset_index(drop=True).copy()
    )
    return straightened_gdf


def postprocess_prediction_to_gdf(input_image, prediction, simplify=True):
    """For a prediction, create a complete floorplan.

    Args:
        input_image (np.array): input b&w image
        prediction (np.array): generator prediction, rgb range should be between 0 and 1.
        simplify (bool): Simplify predicted geodataframe before processing.

    Returns:
        Geodataframe with floorplan built for a prediction.
    """
    preprocessed_img = resize_and_pad(input_image, size=(256, 256), pad_color=255)

    prediction[preprocessed_img == 255] = 1

    floorplan_raw_gdf = image_to_geodataframe(prediction * 255)
    floorplan_raw_gdf = floorplan_raw_gdf.loc[floorplan_raw_gdf.area >= 1].copy()
    walls_doors = floorplan_raw_gdf.loc[
        floorplan_raw_gdf["category"].isin(["wall", "door"])
    ]
    rooms = floorplan_raw_gdf.loc[~floorplan_raw_gdf["category"].isin(["wall", "door"])]
    floorplan_raw_gdf = walls_doors.append(rooms)

    if simplify:
        tolerance = 2
        floorplan_raw_gdf.geometry = floorplan_raw_gdf.simplify(tolerance=tolerance)
    return floorplan_raw_gdf


def postprocess_prediction(input_image, prediction, simplify=True):
    """For a prediction, create a complete floorplan.

    Args:
        input_image (np.array): input b&w image
        prediction (np.array): generator prediction, rgb range should be between 0 and 1.
        simplify (bool): Simplify predicted geodataframe before processing.

    Returns:
        Geodataframe with floorplan built for a prediction.
    """
    floorplan_raw_gdf = postprocess_prediction_to_gdf(input_image, prediction)
    preprocessed_img = resize_and_pad(input_image, size=(256, 256), pad_color=255)
    in_img_gdf = image_to_geodataframe(preprocessed_img)
    in_img_gdf.geometry = [in_img_gdf.unary_union] * in_img_gdf.shape[0]
    in_img_gdf = in_img_gdf.head(1).explode().reset_index(drop=True)

    floorplan_raw_gdf = floorplan_raw_gdf.loc[
        ~floorplan_raw_gdf["category"].isin(["wall", "door"])
    ]
    if simplify:
        tolerance = 3
        floorplan_raw_gdf.geometry = floorplan_raw_gdf.simplify(tolerance=tolerance)

    extended_floorplan_gdf = process_floorplan(floorplan_raw_gdf, in_img_gdf)
    cleaned_extended_floorplan_gdf = clear_overlap_floorplan(extended_floorplan_gdf)

    return cleaned_extended_floorplan_gdf
