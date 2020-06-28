from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from fmlwright.core import preprocessing
from shapely import affinity
from tqdm import tqdm

PURPOSES_TO_KEEP = [
    "bijeenkomstfunctie",
    #     'celfunctie',
    #     'gezondheidszorgfunctie',
    "industriefunctie",
    "kantoorfunctie",
    #     'logiesfunctie',
    #     'onderwijsfunctie',
    #     'overige gebruiksfunctie',
    #     'sportfunctie',
    "winkelfunctie",
    "woonfunctie",
]

KEEP_UNKOWN_PURPOSE = False
SUBSAMPLE_100_OVERLAP = 0.0  # There are a lot of these.
MINIMAL_PERCENTAGE_THRESHOLD = 0.1
MAXIMAL_PERCENTAGE_THRESHOLD = 0.9
SAMPLE_SIZE = None


def _find_relevant_categories(value):
    """Check if the row contains relevant categories."""
    if value is None:
        if KEEP_UNKOWN_PURPOSE:
            return True
        else:
            return False
    return True if sum([x in value for x in PURPOSES_TO_KEEP]) else False


def create_output_image(building_footprint, parcel_footprint, file_path):
    """Generate output images for parcel footprints.

    Args:
        building_footprint (gpd.GeodataFrame): gdf with building footprint.
        parcel_footprint (gpd.GeodataFrame): gdf with parcel.
        file_path (str): File path to store the image.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    gpd.overlay(building_footprint, parcel_footprint, how="symmetric_difference").plot(
        ax=ax, color="lightgray"
    )
    parcel_footprint.geometry.exterior.buffer(0.25).plot(ax=ax, color="black")
    building_footprint.plot(ax=ax, color="black")

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
        quality=IMG_QUALITY,
    )
    plt.close()


def create_input_image(parcel_footprint, file_path):
    """Create the input image.

    Args:
        parcel_footprint (gpd.GeodataFrame): gdf with parcel.
        file_path (str): File path to store the image.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    parcel_footprint.plot(ax=ax, color="lightgray")
    parcel_footprint.geometry.exterior.buffer(0.25).plot(ax=ax, color="black")

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
        quality=IMG_QUALITY,
    )
    plt.close()


def join_kadaster_bag_info(kadaster_gdf, bag_gdf):
    """Returns gdf with bag_gdf that are within a kadaster. Based on perceel_id."""
    return gpd.sjoin(bag_gdf, kadaster_gdf, op="within")


def calculate_percentage_overlap(bag_sample, kadaster_sample):
    """Calculate overlap between bag and kadaster geodataframes in percentages.

    Args:
        bag_sample (gpd.GeodataFrame): gdf with bag sample.
        kadaster_sample (gpd.GeodataFrame): gdf with kadaster sample.

    Returns:
        geodataframe with indices and overlap.
    """
    # calculate percentage intersection

    ov_output = gpd.overlay(bag_sample, kadaster_sample, how="intersection")

    percentage_overlap = (
        ov_output.geometry.area / kadaster_sample.geometry.area.values[0]
    )
    percentage_overlap.index = bag_sample.index
    percentage_overlap = percentage_overlap.to_frame().rename(
        columns={0: "percentage_overlap"}
    )
    percentage_overlap.index.name = "bag_index"
    return percentage_overlap


def generate_images(INPUT_DIRECTORY, OUTPUT_DIRECTORY, IMG_QUALITY):
    """Generate images.

    Args:
        INPUT_DIRECTORY (str): directory with original geojson.
        OUTPUT_DIRECTORY (str): directory to store the images.
        IMG_QUALITY (int): Image quality to store the images as.
    """
    kadastralegrens_gdf = gpd.read_file(INPUT_DIRECTORY / "perceel.geojson")
    kadastralegrens_gdf = kadastralegrens_gdf.to_crs(epsg=3395)
    kadastralegrens_gdf["total_area"] = kadastralegrens_gdf.geometry.area
    kadastralegrens_gdf = kadastralegrens_gdf.rename(columns={"gid": "perceel_id"})

    bag_gdf = gpd.read_file(INPUT_DIRECTORY / "bag.shp")
    bag_gdf = bag_gdf.to_crs(epsg=3395)  # lat:lon = 4326, mercador = 3395

    relevant_bag_gdf = bag_gdf.loc[
        bag_gdf["gebruiksdo"].apply(_find_relevant_categories)
    ]
    print(f"{(relevant_bag_gdf.shape[0] / bag_gdf.shape[0] * 100)}% of rows remain.")
    print(relevant_bag_gdf.shape)

    matching_bag_kad_gdf = join_kadaster_bag_info(
        kadastralegrens_gdf[["identificatieLokaalID", "geometry", "total_area"]],
        relevant_bag_gdf,
    )
    matching_bag_kad_gdf["area"] = matching_bag_kad_gdf.geometry.area
    matching_bag_kad_gdf["percentage_overlap"] = (
        matching_bag_kad_gdf["area"] / matching_bag_kad_gdf["total_area"]
    )

    matched_kadastralegrens_gdf = kadastralegrens_gdf.iloc[
        matching_bag_kad_gdf.index_right.unique()
    ].copy()

    subsample_100_overlap = matching_bag_kad_gdf.loc[
        matching_bag_kad_gdf["percentage_overlap"] == 1
    ].copy()
    occurances = subsample_100_overlap.shape[0]
    subsample_100_overlap = subsample_100_overlap.sample(
        int(occurances * SUBSAMPLE_100_OVERLAP)
    )

    subsampled_overlap_output = matching_bag_kad_gdf.loc[
        matching_bag_kad_gdf["percentage_overlap"] <= MAXIMAL_PERCENTAGE_THRESHOLD
    ].copy()
    subsampled_overlap_output = pd.concat(
        [subsampled_overlap_output, subsample_100_overlap]
    )

    subsampled_overlap_output = subsampled_overlap_output.loc[
        subsampled_overlap_output["percentage_overlap"] >= MINIMAL_PERCENTAGE_THRESHOLD
    ].copy()

    if SAMPLE_SIZE is not None:
        subsampled_overlap_output = subsampled_overlap_output.sample(SAMPLE_SIZE)

    filepaths = []
    for i, overlap_row in tqdm(
        subsampled_overlap_output.iterrows(), total=subsampled_overlap_output.shape[0]
    ):

        filename_input = (
            OUTPUT_DIRECTORY
            / "input"
            / f"GID_{overlap_row.gid}#KAD_{overlap_row.index_right}.png"
        )
        filename_output = (
            OUTPUT_DIRECTORY
            / "output"
            / f"GID_{overlap_row.gid}#KAD_{overlap_row.index_right}.png"
        )
        try:

            parcel_footprint = matched_kadastralegrens_gdf.loc[
                matched_kadastralegrens_gdf.index == overlap_row.index_right
            ].copy()
            angle = preprocessing.find_angle(parcel_footprint)
            centroid = parcel_footprint.centroid.tolist()[0]

            building_footprint = gpd.GeoDataFrame(overlap_row.to_frame()).T

            parcel_footprint.geometry = [
                affinity.rotate(parcel_footprint.geometry.tolist()[0], angle)
            ]

            building_footprint.geometry = [
                affinity.rotate(
                    building_footprint.geometry.tolist()[0], angle, centroid
                )
            ]

            create_input_image(
                parcel_footprint=parcel_footprint, file_path=filename_input
            )

            create_output_image(
                parcel_footprint=parcel_footprint,
                building_footprint=building_footprint,
                file_path=filename_output,
            )

            temp_files = {
                "original_file": str(i),
                "input_file": str(filename_input.absolute()),
                "output_file": str(filename_output.absolute()),
            }
            filepaths.append(temp_files)
        except Exception as e:
            print("-" * 50)
            print(f"something went wrong: {i}")
            print(e)
            continue

    results = pd.DataFrame(filepaths)
    results.to_csv(OUTPUT_DIRECTORY / f"index_file.csv", index=False)


if __name__ == "__main__":
    CITY = "leiden"

    INPUT_DIRECTORY = Path("../../../../data/api_crawl/")
    INPUT_DIRECTORY = INPUT_DIRECTORY / CITY

    OUTPUT_DIRECTORY = Path("../../../data/ground_plans/")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIRECTORY = OUTPUT_DIRECTORY / CITY
    (OUTPUT_DIRECTORY / "input").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / "output").mkdir(parents=True, exist_ok=True)
    IMG_QUALITY = 70

    generate_images(INPUT_DIRECTORY, OUTPUT_DIRECTORY, IMG_QUALITY)
