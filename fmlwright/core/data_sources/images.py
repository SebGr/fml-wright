import matplotlib.pyplot as plt


def save_image_gdf(areas_gdf, file_path, img_quality):
    """Save a geodataframe as an image.

    Args:
        areas_gdf (gpd.GeoDataFrame): geodataframe to store. expects a 'colors' column.
        file_path (str): File name and path.
        img_quality (int): quality of image to store.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    areas_gdf.plot(ax=ax, color=areas_gdf["colors"], categorical=True)

    ax.patch.set_facecolor("white")
    ax.patch.set_edgecolor("white")
    fig.patch.set_visible(False)
    ax.axis("off")
    fig.savefig(
        file_path,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="white",
        edgecolor="white",
        quality=img_quality,
    )
    plt.close()
