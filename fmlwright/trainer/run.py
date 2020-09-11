import logging
from pathlib import Path

from fmlwright.core import data_sources
from fmlwright.trainer.models import BiCycleGAN, Pix2Pix

log = logging.getLogger(__name__)


def run(conf):
    """Function to create, compile and train the neural network.

    Args:
        conf: args from the configuration file as well as additional arguments.

    Returns:
        None
    """
    conf["settings"]["storage_location"] = str(
        Path(conf["settings"]["storage_location"])
        / conf["settings"]["category"]
        / conf["nn_structure"]["generator"]["model_type"]
    )
    log.info(
        f"Storage directory has been updated to {conf['settings']['storage_location']}."
    )

    if conf["nn_structure"]["generator"]["model_type"] == "BiCycleGAN":
        log.info("BiCycleGAN model has been selected.")
        log.info("Creating model.")
        Model = BiCycleGAN(conf=conf)
    elif conf["nn_structure"]["generator"]["model_type"] == "Pix2Pix":
        log.info("Pix2Pix model has been selected.")
        log.info("Creating model.")
        Model = Pix2Pix(conf=conf)
    else:
        Model = None
        log.error("Unknown model type has been selected.")
        log.error("Your options are: BiCycleGAN, Pix2Pix.")
        log.error("Exiting...")
        exit()

    log.info("Loading dataset.")
    dataset_directory = Path(conf["settings"]["dataset_directory"])
    index_location = dataset_directory / "images_index.csv"
    dataset_files = [str(_file) for _file in dataset_directory.glob("*.tfrecords")]
    train_dataset = data_sources.load_dataset(
        dataset_location=dataset_files,
        index_location=index_location,
        category=conf["settings"]["category"],
        sample_size=conf["settings"]["dataset_size"],
        sample=conf["settings"]["sample"],
    )

    train_dataset = train_dataset.shuffle(conf["settings"]["buffer_size"]).batch(
        conf["settings"]["batch_size"]
    )

    data_sources.save_yaml(conf, str(Model.result_storage / "config.yaml"))

    Model.train(
        max_n_steps=conf["settings"]["max_n_steps"],
        train_dataset=train_dataset,
        store_only_last_model=conf["settings"]["store_only_last_model"],
    )
