import matplotlib.pyplot as plt


def plot_room_usage(gdf):
    """Plot the room percentages per floorplan.

    Args:
        gdf (gpd.GeoDataFrame): geodataframe of the rooms. expects ['area', 'colors'] columns.

    Returns:
        figure with colors for room usage.
    """
    gdf = gdf.copy()
    if "area" not in gdf:
        gdf["area"] = gdf.geometry.area

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf[["area"]].T.plot.bar(ax=ax, stacked=True, color=gdf["colors"], legend=False)
    plt.axis("off")
    return fig, ax
