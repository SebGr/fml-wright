import argparse
import logging
import random
from pathlib import Path

import pandas as pd
from fmlwright.core import preprocessing, data_sources, labeling
from tqdm import tqdm

log = logging.getLogger(__name__)


def generate_geodataframe_file(input_directory, output_directory, sample=1.0):
    """Generate the geodataframe file based on a number of txt files.

    Args:
        input_directory (str): Input directory containing the txt files.
        output_directory (str): Output directory to store the geojson files.
        sample (float): Sample size to generate.
    """
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)

    output_directory.mkdir(parents=True, exist_ok=False)
    colormap = labeling.get_room_color_map()

    data_files = [x for x in input_directory.glob("**/*.txt")]
    data_files = random.sample(data_files, int(len(data_files) * sample))
    log.info(f"Creating shape file based on {len(data_files)} samples.")

    index_file = []
    geodata_file = []
    for i, _file in tqdm(enumerate(data_files), total=len(data_files)):
        try:
            (
                walls_gdf,
                doors_gdf,
                rooms_gdf,
                special_gdf,
            ) = data_sources.parse_floorplan_txt(_file)

            floorplan_gdf = preprocessing.create_floorplan_gdf(
                walls_gdf, doors_gdf, rooms_gdf, special_gdf, colormap
            )
            floorplan_gdf["original_file"] = _file.absolute()

            room_counts = preprocessing.calculate_room_counts(floorplan_gdf)
            room_counts["original_file"] = str(_file.absolute())

            index_file.append(room_counts)
            geodata_file.append(floorplan_gdf)

            if i % 500 == 0:
                index_df = pd.concat(index_file)
                index_df = index_df.fillna(0)

                all_floorplans_gdf = pd.concat(geodata_file)
                all_floorplans_gdf["original_file"] = all_floorplans_gdf[
                    "original_file"
                ].astype(str)

                all_floorplans_gdf.to_file(
                    str(output_directory / "floorplans.json"), driver="GeoJSON"
                )
                index_df.to_csv(output_directory / "index_floorplans.csv", index=False)
        except Exception as e:
            print("-" * 50)
            print(f"something went wrong: {_file}")
            print(e)
            continue

    index_df = pd.concat(index_file)
    index_df = index_df.fillna(0)

    all_floorplans_gdf = pd.concat(geodata_file)
    all_floorplans_gdf["original_file"] = all_floorplans_gdf["original_file"].astype(
        str
    )

    all_floorplans_gdf.to_file(
        str(output_directory / "floorplans.json"), driver="GeoJSON"
    )
    index_df.to_csv(output_directory / "index_floorplans.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate text to geodataframe dataset."
    )
    parser.add_argument(
        "--input_directory",
        type=str,
        default="../../../data/representation_prediction/",
        metavar="input_directory",
        help="input directory",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="../../../data/all_representation_prediction/",
        metavar="output_directory",
        help="output directory",
    )
    args = parser.parse_args()

    input_directory = Path(args.input_directory)
    output_directory = Path(args.output_directory)

    output_directory.mkdir(parents=True, exist_ok=True)

    generate_geodataframe_file(input_directory, output_directory)
