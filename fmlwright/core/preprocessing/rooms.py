import geopandas as gpd
from fmlwright.core.preprocessing import create_subsection_walls
from shapely.geometry import LineString
from shapely import ops

import logging

log = logging.getLogger(__name__)


def connect_room_category_to_rooms(floors_gdf, areas_gdf):
    """Connect the room category to room geodataframe.

    Args:
        floors_gdf (gpd.GeoDataFrame): Geodataframe.
        areas_gdf (gpd.GeoDataFrame): Geodataframe.
    """
    temp = floors_gdf.copy()
    temp.geometry = temp.centroid
    matches = gpd.sjoin(areas_gdf, temp, op="contains", how="left")[
        ["category", "index_right", "geometry", "dump_1", "dump_2"]
    ]
    matches = matches.rename(columns={"index_right": "floor_index"})
    return matches


def find_rooms(walls_gdf, rooms_gdf):
    """Create room gdf.

    Args:
        walls_gdf (gpd.GeoDataFrame): Geodataframe.
        rooms_gdf (gpd.GeoDataFrame): Geodataframe.
    """
    temp_walls = walls_gdf.copy()
    temp_walls.geometry = walls_gdf.geometry.scale(xfact=1.25, yfact=1.25)
    res = create_subsection_walls(temp_walls)
    all_polygons = gpd.GeoDataFrame(geometry=list(ops.polygonize(res.geometry.values)))
    areas_gdf = connect_room_category_to_rooms(
        floors_gdf=rooms_gdf, areas_gdf=all_polygons
    )
    areas_gdf["area_size"] = areas_gdf.geometry.area
    areas_gdf = areas_gdf.sort_values("area_size", ascending=False)
    areas_gdf = areas_gdf.reset_index(drop=False).rename(
        columns={"index": "polygon_index"}
    )

    areas_gdf = create_subrooms(areas_gdf=areas_gdf, rooms_gdf=rooms_gdf)
    areas_gdf = areas_gdf.sort_values("area_size", ascending=False)
    return areas_gdf


def calculate_room_counts(floorplan_gdf):
    """Calculate number of rooms.

    Args:
        floorplan_gdf (gpd.GeoDataFrame): Geodataframe.
    """
    room_counts = (
        floorplan_gdf["category"].value_counts().to_frame().T.reset_index(drop=True)
    )
    return room_counts


def clear_overlap_floorplan(gdf):
    """Floorplan geodataframes can have overlapping polygons.

    This function clears overlap based
    on the size of the polygons. Smallest polygons are given highest priority.

    Args:
        gdf (gpd.GeoDataFrame): Geodataframe.

    Returns:
        Geodaframe with overlap cleaned.
    """
    gdf["area"] = gdf.area
    gdf = gdf.sort_values(by="area", ascending=True).reset_index(drop=True)

    for i, _ in gdf.iterrows():
        temp_gdf = gdf.iloc[[i]]

        # First find the overlapping indices
        overlapping_gdf = gpd.overlay(gdf.reset_index(), temp_gdf, how="intersection")
        overlapping_idx = overlapping_gdf.loc[overlapping_gdf["index"] != i][
            "index"
        ].values

        # Update overlapping polygons so that the overlapping parts are removed.
        difference_gdf = gpd.overlay(
            gdf.loc[gdf.index.isin(overlapping_idx)], temp_gdf, how="difference"
        )

        gdf.loc[difference_gdf.index, "geometry"] = difference_gdf.geometry.values

    # In case of multipolygons, give the polygons in a multipolygon their own row
    gdf = gdf.explode().reset_index(drop=True)
    # Recalculate areas.
    gdf["area"] = gdf.area
    return gdf


def create_subrooms(areas_gdf, rooms_gdf):
    """In case of overlapping rooms, create subrooms.

    Args:
        areas_gdf (gpd.GeoDataFrame): Geodataframe.
        rooms_gdf (gpd.GeoDataFrame): Geodataframe.
    """
    areas_gdf = areas_gdf.copy()
    areas_gdf = areas_gdf.sort_values("floor_index")
    indices_occuring_multiple_times = (
        areas_gdf["polygon_index"]
        .value_counts()[areas_gdf["polygon_index"].value_counts() > 1]
        .index.values
    )

    if len(indices_occuring_multiple_times) > 0:
        subset = areas_gdf.loc[
            areas_gdf["polygon_index"].isin(indices_occuring_multiple_times)
        ].copy()
        for ind, grp in subset.groupby(["polygon_index"]):
            temp = rooms_gdf.loc[grp["floor_index"].values].copy().head(2)
            # Todo make it handle n number of locs using voronoi.
            temp.geometry = temp.centroid
            line_between = LineString(list(temp.geometry.values))

            if line_between.length == 0:
                log.warning("Two points overlap, moving on.")
                continue
            p_1 = line_between.parallel_offset(200, "left").centroid
            p_2 = line_between.parallel_offset(200, "right").centroid
            orthogonal_line_between = LineString([p_1, p_2])

            res = (
                gpd.GeoDataFrame(
                    geometry=[
                        ops.split(grp.geometry.values[0], orthogonal_line_between)
                    ]
                )
                .explode()
                .reset_index(drop=True)
            )

            matching_gdf = (
                gpd.sjoin(res, temp)
                .rename(columns={"index_right": "floor_index"})
                .sort_values("floor_index")
            )

            areas_gdf.loc[
                areas_gdf["floor_index"].isin(temp.index), "geometry"
            ] = matching_gdf.geometry.values
    areas_gdf["area_size"] = areas_gdf.geometry.area
    return areas_gdf
