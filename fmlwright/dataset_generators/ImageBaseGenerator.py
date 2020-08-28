import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from fmlwright.core import data_sources, labeling

log = logging.getLogger(__name__)


def split(a, n):
    """Split list a in n equal parts."""
    n = min(n, len(a))
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


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

        self.output_directory.mkdir(parents=True, exist_ok=False)
        (self.output_directory / "input").mkdir(parents=True, exist_ok=True)
        (self.output_directory / "output").mkdir(parents=True, exist_ok=True)

        self.irrelevant_index_columns = []

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

    def generate_single_image(self, original_file, relevant_floorplan_gdf, i):
        """Generate a single image pair.

        Args:
            original_file (Path): Original file location.
            relevant_floorplan_gdf (gpd.GeoDataFrame): geopandas dataframe with relevant
                floorplan objects.
            i (int): Current index number.

        Returns:
            (pd.DataFrame): Dataframe containing index information.
        """
        try:
            relevant_floorplan_gdf = relevant_floorplan_gdf.copy()
            relevant_index_df = relevant_floorplan_gdf.drop(
                self.irrelevant_index_columns + ["colors"], axis=1
            ).head(1)

            if relevant_floorplan_gdf.empty:
                log.error(f"{original_file} is empty.")
                return pd.DataFrame()

            input_gdf = self.create_input_gdf(relevant_floorplan_gdf)
            filename_input = self.output_directory / "input" / f"{i}.png"
            data_sources.save_image_gdf(input_gdf, filename_input, self.img_quality)

            output_gdf = self.create_output_gdf(relevant_floorplan_gdf)
            filename_output = self.output_directory / "output" / f"{i}.png"
            data_sources.save_image_gdf(output_gdf, filename_output, self.img_quality)
            n_floorplan_usages = list(
                relevant_index_df.to_dict(orient="index").values()
            )[0]

            categories = labeling.create_floorplan_categories(n_floorplan_usages)
            relevant_index_df["categories"] = [categories]
            relevant_index_df["original_file"] = str(original_file)
            relevant_index_df["input_file"] = str(filename_input)
            relevant_index_df["output_file"] = str(filename_output)
            return relevant_index_df

        except Exception as e:
            log.error(f"Issue with file: {original_file}")
            log.error(f"{e}")
            return pd.DataFrame()

    def generate_single_block(self, gdf, block_number, dataset_size):
        """Generate a group of images for a single block.

        Args:
            gdf (gpd.GeoDataFrame): Geodataframe with floorplan objects.
            block_number (int): Depicts the current dataset block.
            dataset_size (int):  Number of samples per block.
        """
        i = block_number * dataset_size
        index_files = list()
        for original_file, relevant_floorplan_gdf in tqdm(
            gdf.groupby("original_file"),
            total=i + gdf["original_file"].nunique(),
            initial=i,
            leave=False,
        ):
            _single_index = self.generate_single_image(
                original_file=original_file,
                relevant_floorplan_gdf=relevant_floorplan_gdf,
                i=i,
            )
            i += 1
            index_files.append(_single_index)

        index_df = pd.concat(index_files)
        index_df["dataset_block"] = block_number
        index_df.to_csv(
            self.output_directory / f"index_file_{block_number}.csv", index=False
        )

    def run(self, n_jobs=-1, starting_block=0, dataset_size=500):
        """Function that does all the work.

        This function loads the data and runs through all of the samples in order to create the
        relevant images.

        Args:
            n_jobs (int): Number of parallel threads to use.
            starting_block (int): First original file to start at.
            dataset_size (int): Number of samples per block.
        """
        log.info("Loading files...")
        geojson_files = sorted(self.input_directory.glob("*.json"))
        floorplans_gdf = pd.concat([gpd.read_file(_file) for _file in geojson_files])

        index_floorplan = list(self.input_directory.glob("index_floorplans.csv"))
        if len(index_floorplan) == 0:
            log.info(
                "Did not find a main index floorplan file, building from components..."
            )
            index_floorplan = sorted(
                self.input_directory.glob("index_floorplans_*.csv")
            )
            log.info(f"Found {len(index_floorplan)} index block files.")
            index_files = pd.concat([pd.read_csv(_file) for _file in index_floorplan])
        else:
            index_files = pd.read_csv(self.input_directory / "index_floorplans.csv")
        log.info("Files have been loaded.")

        self.irrelevant_index_columns = list(floorplans_gdf.columns)

        log.info("Generating data blocks...")
        n_blocks = int(len(index_files) / dataset_size)

        data_file_blocks = split(index_files, n_blocks)
        dataset_blocks_ids = np.arange(len(data_file_blocks))

        if starting_block != 0:
            data_file_blocks = data_file_blocks[starting_block:]
            dataset_blocks_ids = dataset_blocks_ids[starting_block:]
            log.info(f"Starting at a different block number: {starting_block}.")

        floorplans_gdf["colors"] = list(
            floorplans_gdf[["color_r", "color_g", "color_b"]].values
        )

        all_data = [
            pd.merge(floorplans_gdf, _data_file_block, how="inner", on="original_file")
            for _data_file_block in data_file_blocks
        ]
        log.info(f"Creating {len(all_data)} blocks.")

        Parallel(n_jobs=n_jobs)(
            delayed(self.generate_single_block)(
                gdf=subset_gdf, block_number=block_number, dataset_size=dataset_size
            )
            for subset_gdf, block_number in tqdm(zip(all_data, dataset_blocks_ids,))
        )

        log.info("Combining the separate index files..")
        index_floorplan = sorted(self.output_directory.glob("index_file_*.csv"))
        log.info(f"Found {len(index_floorplan)} index block files.")
        index_files = pd.concat([pd.read_csv(_file) for _file in index_floorplan])
        index_files = index_files.fillna(0)
        index_files.to_csv(self.output_directory / "index_file.csv", index=False)
