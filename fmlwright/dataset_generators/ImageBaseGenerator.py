import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from fmlwright.core import data_sources, labeling

log = logging.getLogger(__name__)


class ImageBaseGenerator:
    """Baseclass to create images from geodata."""

    def __init__(self, input_directory, output_directory, img_quality):
        """Initialize the base class for image generation.

        Args:
            input_directory (str): Directory with geodata.
            output_directory (str): Target directory for tfrecords dataset.
            img_quality (int): Images quality.
        """
        self.input_directory = Path(input_directory)
        self.output_directory = Path(output_directory)
        self.img_quality = img_quality

        self.store_index_per_n = 1000

        self.output_directory.mkdir(parents=True, exist_ok=False)
        (self.output_directory / "input").mkdir(parents=True, exist_ok=True)
        (self.output_directory / "output").mkdir(parents=True, exist_ok=True)

    def create_input_gdf(self, gdf):
        """Create the input geodataframe.

        Args:
            gdf (gpd.GeoDataFrame): Geodataframe of the building floorplan.

        Returns:
            Geodataframe with the input gdf processed to the correct format.
        """
        raise NotImplementedError

    def create_output_gdf(self, gdf):
        """Create the output geodataframe.

        Args:
            gdf (gpd.GeoDataFrame): Geodataframe of the building floorplan.

        Returns:
            Geodataframe with the input gdf processed to the correct format.
        """
        raise NotImplementedError

    def generate_images(self):
        """Function that does all the work.

        This function loads the data and runs through all of the samples in order to create the
        relevant images.
        """
        log.info("Loading files...")
        floorplans_gdf = gpd.read_file(self.input_directory / "floorplans.json")
        index_files = pd.read_csv(self.input_directory / "index_floorplans.csv")
        log.info("Files have been loaded.")

        irrelevant_index_columns = list(floorplans_gdf.columns)

        filepaths = []

        all_data = pd.merge(
            floorplans_gdf, index_files, how="inner", on="original_file"
        )
        all_data["colors"] = list(all_data[["color_r", "color_g", "color_b"]].values)

        i = 0
        for original_file, relevant_floorplan_gdf in tqdm(
            all_data.groupby("original_file"), total=all_data["original_file"].nunique()
        ):
            try:
                relevant_floorplan_gdf = relevant_floorplan_gdf.copy()
                relevant_index_df = relevant_floorplan_gdf.drop(
                    irrelevant_index_columns + ["colors"], axis=1
                ).head(1)

                if relevant_floorplan_gdf.empty:
                    log.error(f"{original_file} is empty")
                    continue

                input_gdf = self.create_input_gdf(relevant_floorplan_gdf)
                filename_input = self.output_directory / "input" / f"{i}.png"
                data_sources.save_image_gdf(input_gdf, filename_input, self.img_quality)

                output_gdf = self.create_output_gdf(relevant_floorplan_gdf)
                filename_output = self.output_directory / "output" / f"{i}.png"
                data_sources.save_image_gdf(
                    output_gdf, filename_output, self.img_quality
                )

                n_floorplan_usages = list(
                    relevant_index_df.to_dict(orient="index").values()
                )[0]
                categories = labeling.create_floorplan_categories(n_floorplan_usages)
                relevant_index_df["categories"] = [categories]
                relevant_index_df["original_file"] = str(original_file)
                relevant_index_df["input_file"] = str(filename_input)
                relevant_index_df["output_file"] = str(filename_output)

                filepaths.append(relevant_index_df)
                i += 1

                if i % self.store_index_per_n == 0:
                    results = pd.concat(filepaths)
                    results.to_csv(
                        self.output_directory / f"index_file.csv", index=False
                    )

            except Exception as e:
                log.error("-" * 50)
                log.error(f"Issue with file: {original_file}")
                log.error(e)
                continue

        results = pd.concat(filepaths)
        results.to_csv(self.output_directory / f"index_file.csv", index=False)
