from pathlib import Path
import logging

import cv2
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from fmlwright.core import preprocessing
from fmlwright.core.data_sources import _bytes_feature, _int64_feature

log = logging.getLogger(__name__)


class DatasetGenerator:
    """Generates the tfrecords datasets based on input/output images."""

    def __init__(self, input_directory, output_directory, img_size=(256, 256)):
        """Initialize the dataSetGenerator.

        Args:
            input_directory (str): Directory with images.
            output_directory (str): Target directory for tfrecords dataset.
            img_size (int, int): Image size.
        """
        self.input_directory = Path(input_directory)
        self.output_directory = Path(output_directory)
        self.img_size = img_size

        self.output_directory.mkdir(parents=True, exist_ok=False)

    def _encode_image(self, img):
        """Encode image as bytes."""
        return tf.compat.as_bytes(cv2.imencode(".jpg", img)[1].tostring())

    def create_train_example(self, image_A, image_B, index_number):
        """Create a single example.

        Args:
            image_A: Input image.
            image_B: Output image.
            index_number (int): Index number from index csv file.

        Returns:
            tensorflow Example.
        """
        feature = {
            "index": _int64_feature(index_number),
            "image_A_raw": _bytes_feature(image_A),
            "image_B_raw": _bytes_feature(image_B),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    def save_block(self, dataset, dataset_block):
        """Save a block of images.

        Args:
            dataset (list): List of training examples.
            dataset_block (int): Block number.
        """
        record_file = self.output_directory / f"images_{dataset_block}.tfrecords"
        with tf.io.TFRecordWriter(str(record_file)) as writer:
            for item in dataset:
                writer.write(item.SerializeToString())

    def generate_single_example(self, row, index_num):
        """Generate a single train example based on the index file.

        Args:
            row (gpd.GeoSeries): Single index row.
            index_num (int): Index number.

        Returns:
            tf.train.Example for a single training example.
        """
        input_image = cv2.imread(
            row.input_file
        )  # no COLOR_BGR2RGB conversion necessary
        input_image = preprocessing.resize_and_pad(
            input_image, self.img_size, pad_color=255
        )
        encoded_input_img = self._encode_image(input_image)

        output_image = cv2.imread(
            row.output_file
        )  # no COLOR_BGR2RGB conversion necessary
        output_image = preprocessing.resize_and_pad(
            output_image, self.img_size, pad_color=255
        )
        encoded_output_img = self._encode_image(output_image)

        train_example = self.create_train_example(
            image_A=encoded_input_img,
            image_B=encoded_output_img,
            index_number=index_num,
        )
        return train_example

    def generate_dataset(self, dataset_size=5000):
        """Function that does all of the work.

        It loads and creates tfrecords datasets. The number of samples in a single file is based
        on the dataset size.

        Args:
            dataset_size (int): number of samples in a single tfrecords file.
        """
        index_file = pd.read_csv(self.input_directory / "index_file.csv")

        dataset = []
        dataset_block = 0
        dataset_block_storage = []
        for i, row in tqdm(index_file.iterrows(), total=index_file.shape[0]):
            try:
                train_example = self.generate_single_example(row, index_num=i)

                dataset.append(train_example)
                dataset_block_storage.append(dataset_block)
                if (i % dataset_size == 0) and (i != 0):
                    self.save_block(dataset_block=dataset_block, dataset=dataset)
                    dataset_block += 1
                    dataset = []

            except Exception as e:
                log.error(f"\nSomething went wrong: {row}")
                log.error(f"\n{e}")
                continue

        self.save_block(dataset_block=dataset_block, dataset=dataset)

        results = index_file.reset_index(drop=False).rename(
            columns={"index": "image_index"}
        )
        results["dataset_block"] = dataset_block_storage
        results.to_csv(self.output_directory / f"images_index.csv", index=False)
