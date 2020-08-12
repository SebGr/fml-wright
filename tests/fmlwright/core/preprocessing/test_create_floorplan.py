from fmlwright.core import data_sources, preprocessing, labeling
import numpy as np


def test_create_floorplan_gdf_rectangle():
    """Test if a single room is generated correctly."""
    rectangle_file = "../../../test_data/rectangle.txt"
    walls_gdf, doors_gdf, rooms_gdf, special_gdf = data_sources.parse_floorplan_txt(
        rectangle_file
    )

    colormap = labeling.get_room_color_map()

    floorplan_gdf = preprocessing.create_floorplan_gdf(
        walls_gdf, doors_gdf, rooms_gdf, special_gdf, colormap
    )

    area_size_bedroom = floorplan_gdf.loc[floorplan_gdf["category"] == "bedroom"][
        "area_size"
    ].tolist()[0]

    assert floorplan_gdf.shape[0] == 5  # 4 walls and a bedroom
    assert area_size_bedroom == 10000


def test_create_floorplan_gdf_different_double_room():
    """Test if a multi-purpose room with different room functions is generated correctly."""
    rectangle_file = "../../../test_data/different_double_purpose_room.txt"
    walls_gdf, doors_gdf, rooms_gdf, special_gdf = data_sources.parse_floorplan_txt(
        rectangle_file
    )

    colormap = labeling.get_room_color_map()

    floorplan_gdf = preprocessing.create_floorplan_gdf(
        walls_gdf, doors_gdf, rooms_gdf, special_gdf, colormap
    )

    bedroom = floorplan_gdf.loc[floorplan_gdf["category"] == "bedroom"].copy()
    bedroom_xy = np.array(bedroom["geometry"].values[0].exterior.coords.xy)
    living_room = floorplan_gdf.loc[floorplan_gdf["category"] == "living_room"].copy()
    living_room_xy = np.array(living_room["geometry"].values[0].exterior.coords.xy)

    bedroom_expected_xy = (
        np.array([0.0, 0.0, 100.0, 0.0]),
        np.array([0.0, 100.0, 0.0, 0.0]),
    )

    living_room_expected_xy = (
        np.array([0.0, 100.0, 100.0, 0.0]),
        np.array([100.0, 100.0, 0.0, 100.0]),
    )

    assert (bedroom_xy == bedroom_expected_xy).all()
    assert (living_room_xy == living_room_expected_xy).all()


def test_create_floorplan_gdf_same_double_room():
    """Test if a multi-purpose room with the same room functions is generated correctly."""
    rectangle_file = "../../../test_data/same_double_purpose_room.txt"
    walls_gdf, doors_gdf, rooms_gdf, special_gdf = data_sources.parse_floorplan_txt(
        rectangle_file
    )

    colormap = labeling.get_room_color_map()

    floorplan_gdf = preprocessing.create_floorplan_gdf(
        walls_gdf, doors_gdf, rooms_gdf, special_gdf, colormap
    )

    bedrooms = floorplan_gdf.loc[floorplan_gdf["category"] == "bedroom"].copy()
    bedroom_1_xy = np.array(bedrooms["geometry"].values[0].exterior.coords.xy)
    bedroom_2_xy = np.array(bedrooms["geometry"].values[1].exterior.coords.xy)

    bedroom_expected_xy = (
        np.array([0.0, 0.0, 100.0, 0.0]),
        np.array([0.0, 100.0, 0.0, 0.0]),
    )

    living_room_expected_xy = (
        np.array([0.0, 100.0, 100.0, 0.0]),
        np.array([100.0, 100.0, 0.0, 100.0]),
    )

    assert (bedroom_1_xy == bedroom_expected_xy).all()
    assert (bedroom_2_xy == living_room_expected_xy).all()
