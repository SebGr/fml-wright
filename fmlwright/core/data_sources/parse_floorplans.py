import geopandas as gpd
import pandas as pd
from fmlwright.core.labeling import LABELS_TO_IGNORE, LINE_LABELS, SPECIAL_LABELS
from shapely.geometry import LineString, Polygon


def _create_rectangle(row, polygon=True):
    """Create a rectangle."""
    bot_left = [row.x_min, row.y_min]
    bot_right = [row.x_max, row.y_min]
    top_left = [row.x_min, row.y_max]
    top_right = [row.x_max, row.y_max]
    if polygon:
        return Polygon([bot_left, bot_right, top_right, top_left])
    else:
        return [bot_left, bot_right, top_right, top_left]


def parse_floorplan_txt(file_path):
    """Parse a single floorplan text.

    Args:
        file_path (str): file location.

    Returns:
        Geodataframe with the txt information processed.
    """
    df = pd.read_csv(file_path, sep="\t", header=None).dropna(axis=1)
    df.columns = ["x_min", "y_min", "x_max", "y_max", "category", "dump_1", "dump_2"]

    df = df.loc[~df["category"].isin(LABELS_TO_IGNORE)].copy()

    df["x_min"] = df["x_min"].astype(int)
    df["y_min"] = df["y_min"].astype(int)
    df["x_max"] = df["x_max"].astype(int)
    df["y_max"] = df["y_max"].astype(int)

    lines_df = df.loc[df["category"].isin(LINE_LABELS)].copy()
    lines_df["start"] = lines_df.apply(lambda row: (row["x_min"], row["y_min"]), axis=1)
    lines_df["end"] = lines_df.apply(lambda row: (row["x_max"], row["y_max"]), axis=1)
    lines_df["geometry"] = [
        LineString(x) for x in zip(lines_df["start"], lines_df["end"])
    ]

    polygons_df = df.loc[~df["category"].isin(LINE_LABELS)].copy()
    polygons_df["geometry"] = polygons_df.apply(_create_rectangle, axis=1)

    result_gdf = gpd.GeoDataFrame(pd.concat([lines_df, polygons_df], axis=0))

    walls_gdf = result_gdf.loc[result_gdf["category"] == "wall"].copy()
    doors_gdf = result_gdf.loc[result_gdf["category"] == "door"].copy()
    rooms_gdf = result_gdf.loc[
        ~result_gdf["category"].isin(LINE_LABELS + SPECIAL_LABELS)
    ].copy()
    special_gdf = result_gdf.loc[result_gdf["category"].isin(SPECIAL_LABELS)].copy()
    return walls_gdf, doors_gdf, rooms_gdf, special_gdf
