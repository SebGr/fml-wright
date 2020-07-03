import logging
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np

import pandas as pd
from fmlwright.core import preprocessing, data_sources, labeling
from tqdm import tqdm

log = logging.getLogger(__name__)


def split(a, n):
    """Split list a in n equal parts."""
    n = min(n, len(a))
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


class GeoDataGenerator:
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
        dataset_gdf["original_file"] = dataset_gdf[
            "original_file"
        ].astype(str)

        dataset_gdf.to_file(
            str(self.output_directory / f"floorplans_{dataset_block}.json"), driver="GeoJSON"
        )

    def generate_single_example(self, filename):
        """Generate a single geodataframe from a text file.

        Args:
            filename (str): File location

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
        index_file = []
        dataset = []
        for i, _file in enumerate(data_files):
            try:
                floorplan_gdf, index_single_example = self.generate_single_example(_file)
                index_single_example['dataset_block'] = dataset_block

                index_file.append(index_single_example)
                dataset.append(floorplan_gdf)

            except Exception as e:
                log.warning(f"\nIssue with file: {_file}")
                log.warning(f"\n{e}")
                continue
        self.save_block(dataset_block=dataset_block, dataset=dataset)
        return index_file

    def generate_dataset(self, dataset_size=5000):
        """Function that does all of the work.

        It loads txt files and creates geojson files. The number of samples in a single file is
        based  on the dataset size.

        Args:
            dataset_size (int): number of samples in a single geojson file.
        """
        data_files = [x for x in self.input_directory.glob("**/*.txt")]
        log.info(f"Creating shape file based on {len(data_files)} samples.")

        n_blocks = int(len(data_files) / dataset_size)
        data_file_blocks = split(data_files, n_blocks)
        dataset_blocks_ids = np.arange(len(data_file_blocks))

        index_file = Parallel(n_jobs=-1)(delayed(self.generate_single_block)(
            data_file_block,
            dataset_block_id
        ) for (data_file_block, dataset_block_id) in tqdm(zip(data_file_blocks,
                                                              dataset_blocks_ids)))

        index_df = pd.concat(index_file)
        index_df = index_df.fillna(0)
        index_df.to_csv(self.output_directory / "index_floorplans.csv", index=False)
