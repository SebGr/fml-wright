import ast
import datetime
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from fmlwright.modeling.utils import PatchGanLabels, LrScheduler

log = logging.getLogger(__name__)


class BaseModel:
    """Base class for GAN models."""

    def __init__(self, conf):
        """Base model initialization.

        Args:
            conf (dict): loaded configuration file.
        """
        self.input_shape = ast.literal_eval(conf["nn_structure"]["input_shape"])
        self.batch_size = conf["settings"]["batch_size"]

        self.decay_steps_start = conf["stabilization"]["lr_decay"]["steps_start"]
        self.lr_decay_method = conf["stabilization"]["lr_decay"]["method"]
        self.label_type = conf["stabilization"]["label_type"]

        if conf["stabilization"]["ttur"]["use"]:
            self.g_lr = conf["stabilization"]["ttur"]["g_lr"]
            self.d_lr = conf["stabilization"]["ttur"]["d_lr"]
            self.lr = -1
        else:
            self.g_lr = -1
            self.d_lr = -1
            self.lr = conf["settings"]["lr"]

        self.save_model_per_n_steps = conf["settings"]["save_model_per_n_steps"]
        self.save_example_per_n_steps = conf["settings"]["save_example_per_n_steps"]
        self.epoch = 0
        self.max_n_steps = conf["settings"]["max_n_steps"]
        self.steps = 0

        current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.result_storage = (
            Path(conf["settings"]["storage_location"]) / current_datetime
        )
        log.info(f"Storage location: {self.result_storage}")
        self.image_storage = self.result_storage / "images"
        self.image_storage.mkdir(exist_ok=True, parents=True)

        self.summary_writer = tf.summary.create_file_writer(
            str(
                Path(conf["settings"]["storage_location"])
                / "tensorboard"
                / current_datetime
            )
        )

        self.ttur = conf["stabilization"]["ttur"]["use"]
        self.update_lr_every_n_steps = conf["stabilization"]["lr_decay"][
            "update_every_n_steps"
        ]

        self.num_D = 1
        self.patch_size = 30
        self.second_patch_size = 14

        self.add_noise_disc_input = False
        self.n_noise_disc_input = 0

        self.patch_label_maker = PatchGanLabels(self.label_type)
        self.lr_optimizer_scheduler = LrScheduler(
            lr_decay_method=self.lr_decay_method,
            max_n_steps=self.max_n_steps,
            decay_steps_start=self.decay_steps_start,
        )

        self.generator_optimizers = []
        self.disc_optimizers = []

    def save_models(self, models_directory, version=None):
        """Save the models, required to be overwritten by GAN model.

        Args:
            models_directory (Path): main directory to store the models.
            version (int): Model version, depending on stage in training.
        """
        raise NotImplementedError

    def create_example(self, batch):
        """Create an example, required to be overwritten by GAN model.

        Args:
            batch (tf batch): tensorflow batch.
        """
        raise NotImplementedError

    def batch_train(self, batch, current_step, disc_std):
        """Batch train the model.

        Args:
            batch (tensor): tensor with batches of real input and target.
            current_step (tensor): tensor with int value depicting current step.
            disc_std (tensor): tensor with float value depicting current std for disc noise.
        """
        raise NotImplementedError

    def create_label(self, true_label):
        """Create the label set for the discriminator.

        Args:
            true_label (bool): True or False depending labels.

        Returns:
            np array with relevant values.
        """
        disc_patch = [self.batch_size, self.patch_size, self.patch_size, 1]

        label = self.patch_label_maker.create_labels(disc_patch, true_label)
        return label

    def _assert_validity_batch(self, batch):
        """Assert validity of the batch."""
        if batch[0].shape[0] != self.batch_size:
            log.warning("Invalid batch size, skipping this batch..")
            return False
        return True

    def update_learning_rate(self):
        """Update the learning rate for the models."""
        if self.ttur:
            self.lr_optimizer_scheduler.update_learning_rate(
                step=self.steps, lr=self.g_lr, optimizers=self.generator_optimizers
            )

            self.lr_optimizer_scheduler.update_learning_rate(
                step=self.steps, lr=self.d_lr, optimizers=self.disc_optimizers
            )
        else:
            self.lr_optimizer_scheduler.update_learning_rate(
                step=self.steps,
                lr=self.lr,
                optimizers=self.disc_optimizers + self.generator_optimizers,
            )

    def calculate_disc_noise(self, start_noise=0.1):
        """Calculate discriminator noise.

        Args:
            start_noise (float): Starting noise.

        Returns:
            Standard deviation for noise.
        """
        if self.add_noise_disc_input:
            return tf.convert_to_tensor(
                max(
                    0.0,
                    start_noise
                    - (start_noise * (self.steps / self.n_noise_disc_input)),
                )
            )
        else:
            return tf.convert_to_tensor(0.0)

    def train(self, max_n_steps, train_dataset, store_only_last_model=True):
        """Overarching train function, runs through the epochs.

        Args:
            max_n_steps (int): Maximum number of steps the model should run.
            train_dataset (tf dataset): Dataset with examples and labels.
            store_only_last_model (bool): Whether to store just the last model, or also all
            models in between. Storing models in between will take a lot of storage.
        """
        batch_amount = len([x for x in train_dataset])
        self.max_n_steps = max_n_steps + 1
        self.steps = 0
        self.epoch = 0

        n_epochs = int(np.ceil(max_n_steps / batch_amount))
        log.info(f"Running for {n_epochs} epochs.")

        for epoch in tqdm(range(n_epochs)):

            disc_std = self.calculate_disc_noise()

            for i, batch_data in tqdm(
                enumerate(train_dataset), total=batch_amount, leave=False
            ):
                if self._assert_validity_batch(batch_data):
                    self.steps += 1
                    cur_step = tf.convert_to_tensor(self.steps, dtype=tf.int64)
                    self.batch_train(batch_data, cur_step, disc_std)

                if np.floor(self.steps % self.update_lr_every_n_steps) == 0:
                    self.update_learning_rate()

                if np.floor(self.steps % self.save_example_per_n_steps) == 0:
                    self.create_example(train_dataset.take(1))

                if np.floor(self.steps % self.save_model_per_n_steps) == 0:
                    if store_only_last_model:
                        self.save_models(self.result_storage / "models", version=None)
                    else:
                        self.save_models(
                            self.result_storage / "models", version=self.steps
                        )

                if self.steps > self.max_n_steps:
                    break

            self.epoch = epoch

        if store_only_last_model:
            self.save_models(self.result_storage / "models", version=None)
        else:
            self.save_models(self.result_storage / "models", version=self.steps)

        log.info("Finished training, shutting down...")
