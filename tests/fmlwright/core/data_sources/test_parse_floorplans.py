from fmlwright.core import data_sources


def test_parse_floorplan_txt_rectangle():
    """Test if a floorplan txt file is parsed correctly."""
    rectangle_file = "./tests/test_data/rectangle.txt"
    walls_gdf, doors_gdf, rooms_gdf, special_gdf = data_sources.parse_floorplan_txt(
        rectangle_file
    )

    assert walls_gdf.shape[0] == 4  # 4 walls
    assert rooms_gdf.shape[0] == 1  # 1 bedroom
    assert "bedroom" in rooms_gdf["category"].values


def test_parse_floorplan_txt_no_rooms_gdf():
    """Test if a situation with no rooms is handled correctly."""
    walls_file = "./tests/test_data/walls.txt"
    walls_gdf, doors_gdf, rooms_gdf, special_gdf = data_sources.parse_floorplan_txt(
        walls_file
    )

    assert rooms_gdf.empty
    assert walls_gdf.shape[0] == 4
