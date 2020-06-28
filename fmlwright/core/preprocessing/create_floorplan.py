import pandas as pd

from fmlwright.core import preprocessing


def create_floorplan_gdf(walls_gdf, doors_gdf, rooms_gdf, special_gdf, colormap):
    """Create the floorplan GeoDataframe.

    Args:
        walls_gdf:
        doors_gdf:
        rooms_gdf:
        special_gdf:
        colormap:

    Returns:
        pandas dataframe containing shapely geometries of the areas.
    """
    areas_gdf = preprocessing.find_rooms(walls_gdf, rooms_gdf)

    walls_gdf = walls_gdf[["geometry"]].copy()
    walls_gdf["category"] = "wall"

    doors_gdf = doors_gdf[["geometry"]].copy()
    doors_gdf["category"] = "door"

    areas_gdf = pd.concat(
        [areas_gdf, special_gdf[["category", "geometry"]], walls_gdf, doors_gdf], axis=0
    )

    areas_gdf.loc[areas_gdf["category"].isna(), "category"] = "unknown"
    areas_gdf = areas_gdf.reset_index(drop=True)

    areas_gdf = areas_gdf.loc[
        areas_gdf["category"] != "point"
    ].copy()  # todo: fix this ugly hack
    areas_gdf = areas_gdf.reset_index(drop=True)
    colors = pd.DataFrame(
        list(
            areas_gdf["category"]
            .apply(lambda cat: [x / 255 for x in colormap[cat]])
            .values
        ),
        columns=["color_r", "color_g", "color_b"],
    )

    areas_gdf = pd.concat([areas_gdf, colors], axis=1)

    return areas_gdf[
        ["category", "area_size", "color_r", "color_g", "color_b", "geometry"]
    ]
