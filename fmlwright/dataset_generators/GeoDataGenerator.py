import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from fmlwright.core import preprocessing, data_sources, labeling

log = logging.getLogger(__name__)


def split(a, n):
    """Split list a in n equal parts."""
    n = min(n, len(a))
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


class GeoDataGenerator:
    """The GeoDataFrame generator class that turns txt files into geojson."""

    def __init__(self, input_directory, output_directory):
        """Initialize the GeoDataGenerator.

        Args:
            input_directory (str): Directory with images.
            output_directory (str): Target directory for geojson files.
        """
        self.input_directory = Path(input_directory)
        self.output_directory = Path(output_directory)

        self.output_directory.mkdir(parents=True, exist_ok=False)

    def save_block(self, dataset, dataset_block):
        """Save a block of images.

        Args:
            dataset (list): List of geojson items.
            dataset_block (int): Block number.
        """
        dataset_gdf = pd.concat(dataset)
        dataset_gdf["original_file"] = dataset_gdf["original_file"].astype(str)

        dataset_gdf.to_file(
            str(self.output_directory / f"floorplans_{dataset_block}.json"),
            driver="GeoJSON",
        )

    def generate_single_example(self, filename):
        """Generate a single geodataframe from a text file.

        Args:
            filename (Path): File location

        Returns:
            Tuple with the geodataframe and a dictionary with index information.
        """
        (
            walls_gdf,
            doors_gdf,
            rooms_gdf,
            special_gdf,
        ) = data_sources.parse_floorplan_txt(filename)
        colormap = labeling.get_room_color_map()

        floorplan_gdf = preprocessing.create_floorplan_gdf(
            walls_gdf, doors_gdf, rooms_gdf, special_gdf, colormap
        )
        floorplan_gdf["original_file"] = filename.absolute()

        index_single_example = preprocessing.calculate_room_counts(floorplan_gdf)
        index_single_example["original_file"] = str(filename.absolute())
        return floorplan_gdf, index_single_example

    def generate_single_block(self, data_files, dataset_block):
        """Generate the results for a single block.

        Args:
            data_files (list): List of data files.
            dataset_block (int): Depicts the current dataset block.

        Returns:
            Index files
        """
        index_file = list()
        dataset = list()
        for i, _file in enumerate(data_files):
            try:
                floorplan_gdf, index_single_example = self.generate_single_example(
                    _file
                )
                index_single_example["dataset_block"] = dataset_block
                index_file.append(index_single_example)
                dataset.append(floorplan_gdf)

            except Exception as e:
                log.error(f"Issue with file: {_file}: {e}")
                continue
        self.save_block(dataset_block=dataset_block, dataset=dataset)

        index_df = pd.concat(index_file)
        index_df = index_df.fillna(0)
        index_df.to_csv(
            self.output_directory / f"index_floorplans_{dataset_block}.csv", index=False
        )

    def run(self, dataset_size=1000, n_jobs=-1, starting_block=0):
        """Function that does all of the work.

        It loads txt files and creates geojson files. The number of samples in a single file is
        based  on the dataset size.

        Args:
            dataset_size (int): Number of samples in a single geojson file.
            n_jobs (int): Number of parallel threads to use.
            starting_block (int): Starting block to continue the image generation.
        """
        data_files = sorted(self.input_directory.glob("**/*.txt"))
        log.info(f"Creating shape file based on {len(data_files)} samples.")

        n_blocks = int(len(data_files) / dataset_size)
        data_file_blocks = split(data_files, n_blocks)
        dataset_blocks_ids = np.arange(len(data_file_blocks))

        if starting_block != 0:
            data_file_blocks = data_file_blocks[starting_block:]
            dataset_blocks_ids = dataset_blocks_ids[starting_block:]
            log.info(f"Starting at a different block number: {starting_block}.")

        log.info(f"Going through {n_blocks} blocks in parallel.")
        Parallel(n_jobs=n_jobs)(
            delayed(self.generate_single_block)(data_file_block, dataset_block_id)
            for (data_file_block, dataset_block_id) in tqdm(
                zip(data_file_blocks, dataset_blocks_ids)
            )
        )

        log.info("Combining the separate index files..")
        index_floorplan = sorted(self.output_directory.glob("index_floorplans_*.csv"))
        log.info(f"Found {len(index_floorplan)} index block files.")
        index_files = pd.concat([pd.read_csv(_file) for _file in index_floorplan])
        index_files = index_files.fillna(0)
        index_files.to_csv(self.output_directory / "index_floorplans.csv", index=False)
