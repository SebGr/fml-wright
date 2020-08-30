import logging

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import (
    MeanAbsoluteError,
    MeanSquaredError,
    BinaryCrossentropy,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from fmlwright.modeling.models.BaseModel import BaseModel
from fmlwright.modeling.neural_networks.networks import (
    create_discriminator,
    create_generator,
)

log = logging.getLogger(__name__)


class Pix2Pix(BaseModel):
    """Pix2Pix model."""

    def __init__(self, conf):
        """Initialize the Pix2Pix model.

        Args:
            conf file (dict): loaded configuration file.
        """
        super().__init__(conf)

        conf_generator = conf["nn_structure"]["generator"]
        self.G = create_generator(conf_generator, self.input_shape, 0)

        conf_discriminator = conf["nn_structure"]["discriminator"]
        self.D = create_discriminator(conf_discriminator, self.input_shape)

        self.disc_loss_function = (
            BinaryCrossentropy(from_logits=True)
            if conf_discriminator["loss_function"] == "BCE"
            else MeanSquaredError()
        )

        # loss_coeffs
        self.DM_loss_coeff = conf["loss_coeffs"]["DM_loss_coeff"]
        self.L1_loss_coeff = conf["loss_coeffs"]["L1_loss_coeff"]
        self.gan_loss_coeff = conf["loss_coeffs"]["gan_loss_coeff"]

        # Optimizers
        if self.ttur:
            self.D_optimizer = Adam(learning_rate=self.d_lr, beta_1=0.5)
            self.G_optimizer = Adam(learning_rate=self.g_lr, beta_1=0.5)
        else:
            self.D_optimizer = Adam(learning_rate=self.lr, beta_1=0.5)
            self.G_optimizer = Adam(learning_rate=self.lr, beta_1=0.5)

        self.disc_optimizers = [self.D_optimizer]
        self.generator_optimizers = [self.G_optimizer]

    def load_models(self, models_directory, version=None):
        """Load all models.

        Args:
            models_directory (Path): Root directory of models.
            version (int): version number.
        """
        version = "_" + str(version) if version else ""
        self.D = load_model(models_directory / f"discriminator{version}.h5")
        self.G = load_model(models_directory / f"generator{version}.h5")

    def save_models(self, models_directory, version=None):
        """Save the model weights.

        Args:
            models_directory (Path): Root directory of models.
            version (int): version number. For these models it's the current epoch number.
        """
        models_directory.mkdir(parents=True, exist_ok=True)
        version = "_" + str(version) if version else ""
        self.D.save(models_directory / f"discriminator{version}.h5")
        self.G.save(models_directory / f"generator{version}.h5")

    def calculate_D_loss(self, discriminator_b_real, discriminator_b_false):
        """Calculate the D loss.

        Args:
            discriminator_b_real (tf.tensor): Discriminator prediction for real B image.
            discriminator_b_false (tf.tensor): Discriminator prediction for generated B image.

        Returns:
            tensors containing D true and false loss.
        """
        D_true_loss = (
            self.disc_loss_function(self.create_label(True), discriminator_b_real)
            * self.DM_loss_coeff
        )
        D_false_loss = (
            self.disc_loss_function(self.create_label(False), discriminator_b_false)
            * self.DM_loss_coeff
        )

        return D_true_loss, D_false_loss

    def calculate_G_loss(self, discriminator_b_false, real_target, generated_b):
        """Calculate the G loss.

        Args:
            discriminator_b_false (tf.tensor): Discriminator prediction for generated B image.
            real_target (tf.tensor): Real B images.
            generated_b (tf.tensor): Generated B image.

        Returns:
            Tensors containing gan loss and l1 loss.
        """
        MAE = MeanAbsoluteError()
        gan_loss = (
            self.disc_loss_function(self.create_label(True), discriminator_b_false)
            * self.gan_loss_coeff
        )
        l1_loss = MAE(generated_b, real_target) * self.L1_loss_coeff
        return gan_loss, l1_loss

    @tf.function
    def batch_train(self, batch, current_step, disc_std):
        """Batch train the model.

        Args:
            batch (tensor): tensor with batches of real input and target.
            current_step (tensor): tensor with int value depicting current step.
            disc_std (tensor): tensor with float value depicting current std for disc noise.
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            real_input, real_target = batch
            disc_input_noise = tf.random.normal(
                mean=0, stddev=disc_std, shape=self.input_shape
            )

            generated_b = self.G(real_input, training=True)

            discriminator_b_false = self.D(
                generated_b + disc_input_noise, training=True
            )
            discriminator_b_real = self.D(real_target + disc_input_noise, training=True)

            # D losses
            D_true_loss, D_false_loss = self.calculate_D_loss(
                discriminator_b_real, discriminator_b_false
            )

            # G_E loss
            gan_loss, l1_loss = self.calculate_G_loss(
                discriminator_b_false, real_target, generated_b
            )

            total_D_loss = D_true_loss + D_false_loss
            total_G_loss = gan_loss + l1_loss

        generator_gradients = gen_tape.gradient(
            total_G_loss, self.G.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            total_D_loss, self.D.trainable_variables
        )
        self.G_optimizer.apply_gradients(
            zip(generator_gradients, self.G.trainable_variables)
        )
        self.D_optimizer.apply_gradients(
            zip(discriminator_gradients, self.D.trainable_variables)
        )

        with self.summary_writer.as_default():
            tf.summary.scalar("model/total_D_loss", total_D_loss, step=current_step)
            tf.summary.scalar("model/total_G_loss", total_G_loss, step=current_step)
            tf.summary.scalar("D/D_true_loss", D_true_loss, step=current_step)
            tf.summary.scalar("D/D_random_loss", D_false_loss, step=current_step)
            tf.summary.scalar("G/gan_loss", gan_loss, step=current_step)
            tf.summary.scalar("G/l1_loss", l1_loss, step=current_step)
            tf.summary.scalar(
                "model_info/G_lr", self.G_optimizer.learning_rate, step=current_step
            )
            tf.summary.scalar(
                "model_info/D_lr", self.D_optimizer.learning_rate, step=current_step
            )

    def create_example(self, example):
        """Creates and stores four examples based on a random input image.

        Args:
            example (tf batch): tensorflow batch.
        """
        for input_image, output_image in example:
            predicted_image = self.G.predict(input_image.numpy())
            plt.figure(figsize=(15, 15))

            display_list = [
                (input_image[0].numpy() + 1) * 0.5,
                (output_image[0] + 1) * 0.5,
                (predicted_image[0] + 1) * 0.5,
            ]
            title = ["Input Image", "Ground Truth", "Predicted Image"]
            for i in range(3):
                plt.subplot(1, 3, i + 1)
                plt.title(title[i])
                plt.imshow(display_list[i])
                plt.axis("off")

            plt.tight_layout()
            plt.savefig(str(self.image_storage / f"example_steps_{self.steps}"))
            plt.close()
