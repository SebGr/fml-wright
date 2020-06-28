import ast
import logging

import pandas as pd
import tensorflow as tf

log = logging.getLogger(__name__)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_index_subset(index_file, category):
    """Filter an index based on category.

    Args:
        index_file (pd.DataFrame): Dataframe with dataframe index information.
        category (str): category to filter on.

    Returns:
        Dataframe with the subset of index rows.
    """
    cleaned_index_file = index_file.drop(
        ["input_file", "original_file"], axis=1
    ).set_index("output_file")

    cleaned_index_file["categories"] = cleaned_index_file["categories"].apply(
        lambda x: ast.literal_eval(x)
    )

    categories_index_file = pd.DataFrame(
        list(cleaned_index_file["categories"].values), index=cleaned_index_file.index
    )

    if category.startswith("no_"):
        category = "_".join(category.split("no_")[1:])
        categories_index_file = categories_index_file.loc[
            ~categories_index_file[category]
        ]
    else:
        categories_index_file = categories_index_file.loc[
            categories_index_file[category]
        ]
    return index_file.loc[
        index_file["output_file"].isin(categories_index_file.index)
    ].reset_index()


def normalize_image(image):
    """Normalize an image between 1 and -1.

    Args:
        image: original image.

    Returns:
        Normalized tensor of image between 1 and -1
    """
    return tf.cast(image, tf.float32) / 127.5 - 1


def decode_and_normalize(example):
    """Decode and normalize a tf.Example.

    Args:
        example (tf.Example): Example to decode and normalize.

    Returns:
        input and output images, decoded and normalized between 1 and -1.
    """
    image_A = example["image_A_raw"]
    image_A = tf.image.decode_jpeg(image_A)
    image_A = normalize_image(image_A)

    image_B = example["image_B_raw"]
    image_B = tf.image.decode_jpeg(image_B)
    image_B = normalize_image(image_B)

    return image_A, image_B


def filter_images(example, idx_to_keep):
    """Filter examples based on idx.

    Args:
        example (tf.Example): image example.
        idx_to_keep (list): List of indices to keep.

    Returns:
        example or non depending on whether the image is in idx_to_keep.
    """
    if example["index"].numpy() in idx_to_keep:
        return example
    else:
        return None


def _parse_image_function(example_proto, feature):
    """Parse an image base on the features."""
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature)


def load_dataset(dataset_location, index_location, category, dataset_size=1500):
    """Load the dataset.

    Args:
        dataset_location (str): Folder where the dataset is.
        index_location (str): file location of index files.
        category (str): Category to keep.
        dataset_size (int): Sample size of dataset to use.

    Returns:
        tf.dataset with examples.
    """
    dataset = tf.data.TFRecordDataset(dataset_location)

    index_file = pd.read_csv(index_location)
    log.info(f"Index file has {index_file.shape[0]} rows.")
    index_subset = create_index_subset(index_file, category)
    log.info(f"Index subset has {index_subset.shape[0]} rows.")
    if index_subset.shape[0] > dataset_size:
        index_subset = index_subset.sample(dataset_size)
    log.info(
        f"Sampling a maximum of {dataset_size} examples; left with {index_subset.shape[0]} samples."
    )

    feature = {
        "index": tf.io.FixedLenFeature([], tf.int64),
        "image_A_raw": tf.io.FixedLenFeature([], tf.string),
        "image_B_raw": tf.io.FixedLenFeature([], tf.string),
    }

    idx_to_keep = index_subset["index"].values

    parsed_images_A = []
    parsed_images_B = []
    for example in dataset:
        tf_example = _parse_image_function(example, feature)
        if tf_example["index"].numpy() in idx_to_keep:
            image_A, image_B = decode_and_normalize(tf_example)
            parsed_images_A.append(image_A)
            parsed_images_B.append(image_B)

    log.info(f"Dataset created with {len(parsed_images_A)} samples.")
    parsed_dataset = tf.data.Dataset.from_tensor_slices(
        (parsed_images_A, parsed_images_B)
    )
    return parsed_dataset
