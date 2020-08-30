import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import (
    MeanSquaredError,
    BinaryCrossentropy,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from fmlwright.core.utils import create_z_random
from fmlwright.modeling.models import BaseModel
from fmlwright.modeling.neural_networks.networks import (
    create_discriminator,
    create_generator,
    create_encoder,
)

log = logging.getLogger(__name__)


class BiCycleGAN(BaseModel):
    """Generate the BiCycleGAN class."""

    def __init__(self, conf):
        """Initialize the BiCycleGAN.

        Args:
            conf file (dict): loaded configuration file.
        """
        super().__init__(conf)
        self.latent_vector = conf["nn_structure"]["latent_vector"]

        # loss_coeffs
        self.DM_loss_coeff = conf["loss_coeffs"]["DM_loss_coeff"]
        self.L1_loss_coeff = conf["loss_coeffs"]["L1_loss_coeff"]
        self.KL_loss_coeff = conf["loss_coeffs"]["KL_loss_coeff"]
        self.LRM_loss_coeff = conf["loss_coeffs"]["LRM_loss_coeff"]
        self.gan_loss_coeff = conf["loss_coeffs"]["gan_loss_coeff"]
        self.vaegan_loss_coeff = conf["loss_coeffs"]["vaegan_loss_coeff"]

        self.add_noise_disc_input = conf["stabilization"]["discriminator_noise_input"][
            "add"
        ]
        self.n_noise_disc_input = conf["stabilization"]["discriminator_noise_input"][
            "max_n_steps"
        ]

        conf_generator = conf["nn_structure"]["generator"]
        self.G = create_generator(conf_generator, self.input_shape, self.latent_vector)

        conf_discriminator = conf["nn_structure"]["discriminator"]
        self.D = create_discriminator(conf_discriminator, self.input_shape)

        self.num_D = conf_discriminator["num_D"]
        self.disc_loss_function = (
            BinaryCrossentropy(from_logits=True)
            if conf_discriminator["loss_function"] == "BCE"
            else MeanSquaredError()
        )

        conf_encoder = conf["nn_structure"]["encoder"]
        self.E = create_encoder(conf_encoder, self.input_shape, self.latent_vector)

        if self.ttur:
            self.D_optimizer = Adam(learning_rate=self.d_lr, beta_1=0.5)
            self.G_optimizer = Adam(learning_rate=self.g_lr, beta_1=0.5)
            self.E_optimizer = Adam(learning_rate=self.g_lr, beta_1=0.5)
        else:
            self.D_optimizer = Adam(learning_rate=self.lr, beta_1=0.5)
            self.G_optimizer = Adam(learning_rate=self.lr, beta_1=0.5)
            self.E_optimizer = Adam(learning_rate=self.lr, beta_1=0.5)

        self.disc_optimizers = [self.D_optimizer]
        self.generator_optimizers = [self.G_optimizer, self.E_optimizer]

    def calculate_G_E_loss(
        self,
        discriminator_b_random,
        discriminator_b_encoded,
        generated_b_encoded,
        real_target,
        z_enc_log_sigma,
        z_enc_mu,
    ):
        """Calculate G_E loss.

        Args:
            discriminator_b_random (tf.tensor): Discriminator prediction for generated B image
                with z_random.
            discriminator_b_encoded (tf.tensor): Discriminator prediction for generated B image
                with z_encoded.
            generated_b_encoded (tf.tensor): Generated B image with z_encoded.
            real_target (tf.tensor): Real B images.
            z_enc_log_sigma (tf.tensor): z_encoded log_sigma values.
            z_enc_mu (tf.tensor): z_encoded mu values.

        Returns:
            tensors with the G_E losses.
        """
        gan_loss = (
            self.disc_loss_function(self.create_label(True), discriminator_b_random)
            * self.gan_loss_coeff
        )
        vaegan_loss = (
            self.disc_loss_function(self.create_label(True), discriminator_b_encoded)
            * self.vaegan_loss_coeff
        )
        l1_loss = (
            tf.reduce_mean(tf.abs(real_target - generated_b_encoded))
            * self.L1_loss_coeff
        )
        kl_loss = tf.reduce_sum(
            1 + z_enc_log_sigma - z_enc_mu ** 2 - tf.exp(z_enc_log_sigma)
        ) * (-0.5 * self.KL_loss_coeff)

        return gan_loss, vaegan_loss, l1_loss, kl_loss

    def calculate_G_loss(self, z_rand_mu, z_random):
        """Calculate the G loss.

        This is the latent loss.
        z -> B' -> z_mu

        Args:
            z_rand_mu (tf.tensor): z_random mu values.
            z_random (tf.tensor): z_random values.

        Returns:
            Tensor with a specific part of just the G loss.
        """
        clr_loss = tf.reduce_mean(tf.abs(z_rand_mu - z_random)) * self.LRM_loss_coeff
        return clr_loss

    def calculate_D_loss(
        self, discriminator_b_real, discriminator_b_random, discriminator_b_encoded
    ):
        """Calculate the D loss.

        Args:
            discriminator_b_real (tf.tensor): Discriminator prediction for real B image.
            discriminator_b_random (tf.tensor): Discriminator prediction for generated B image
                with z_random.
            discriminator_b_encoded (tf.tensor): Discriminator prediction for generated B image
                with z_encoded.

        Returns:
            tensors with the D loss parts.
        """
        D_true_loss = (
            self.disc_loss_function(self.create_label(True), discriminator_b_real)
            * self.DM_loss_coeff
        )

        D_random_loss = (
            self.disc_loss_function(discriminator_b_random, self.create_label(False))
            * self.DM_loss_coeff
        )
        D_encoded_loss = (
            self.disc_loss_function(discriminator_b_encoded, self.create_label(False))
            * self.DM_loss_coeff
        )
        return D_true_loss, D_random_loss, D_encoded_loss

    @tf.function
    def train_D(self, batch, disc_std):
        """Batch train the discriminator.

        Args:
            batch (tensor): tensor with batches of real input and target.
            disc_std (tensor): tensor with float value depicting current std for disc noise.

        Returns:
            Tensors with D losses.
        """
        disc_input_noise = tf.random.normal(
            mean=0, stddev=disc_std, shape=self.input_shape
        )

        with tf.GradientTape() as disc_tape:
            real_input, real_target = batch
            z_random = tf.random.normal(shape=[self.batch_size, self.latent_vector])

            # cVAE GAN
            # B -> z' -> B'
            z_encoded, z_enc_mu, z_enc_log_sigma = self.E(real_target, training=True)
            generated_b_encoded = self.G([real_input, z_encoded], training=True)

            # cLR GAN
            # z -> B' -> z'
            generated_b_random = self.G([real_input, z_random], training=True)
            _, z_rand_mu, _ = self.E(generated_b_random, training=True)

            discriminator_b_random = self.D(
                generated_b_random + disc_input_noise, training=True
            )
            discriminator_b_encoded = self.D(
                generated_b_encoded + disc_input_noise, training=True
            )
            discriminator_b_real = self.D(real_target + disc_input_noise, training=True)

            # D losses
            D_true_loss, D_random_loss, D_encoded_loss = self.calculate_D_loss(
                discriminator_b_real, discriminator_b_random, discriminator_b_encoded
            )

            total_D_loss = D_true_loss + D_true_loss + D_random_loss + D_encoded_loss

        discriminator_gradients = disc_tape.gradient(
            total_D_loss, self.D.trainable_variables
        )
        self.D_optimizer.apply_gradients(
            zip(discriminator_gradients, self.D.trainable_variables)
        )

        return total_D_loss, D_true_loss, D_random_loss, D_encoded_loss

    @tf.function
    def train_G_E(self, batch, disc_std):
        """Batch train the generator and encoder.

        Args:
            batch (tensor): tensor with batches of real input and target.
            disc_std (tensor): tensor with float value depicting current std for disc noise.

        Returns:
            tensors with the G and E losses.
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as enc_tape:
            real_input, real_target = batch
            z_random = tf.random.normal(shape=[self.batch_size, self.latent_vector])
            disc_input_noise = tf.random.normal(
                mean=0, stddev=disc_std, shape=self.input_shape
            )

            # cVAE GAN
            # B -> z' -> B'
            z_encoded, z_enc_mu, z_enc_log_sigma = self.E(real_target, training=True)
            generated_b_encoded = self.G([real_input, z_encoded], training=True)

            # cLR GAN
            # z -> B' -> z'
            generated_b_random = self.G([real_input, z_random], training=True)
            _, z_rand_mu, _ = self.E(generated_b_random, training=True)

            discriminator_b_random = self.D(
                generated_b_random + disc_input_noise, training=True
            )
            discriminator_b_encoded = self.D(
                generated_b_encoded + disc_input_noise, training=True
            )

            # G_E loss
            gan_loss, vaegan_loss, l1_loss, kl_loss = self.calculate_G_E_loss(
                discriminator_b_random,
                discriminator_b_encoded,
                generated_b_encoded,
                real_target,
                z_enc_log_sigma,
                z_enc_mu,
            )

            # G_loss
            clr_loss = self.calculate_G_loss(z_rand_mu, z_random)

            total_G_loss = gan_loss + vaegan_loss + l1_loss + kl_loss + clr_loss
            total_E_loss = gan_loss + vaegan_loss + l1_loss + kl_loss

        generator_gradients = gen_tape.gradient(
            total_G_loss, self.G.trainable_variables
        )
        self.G_optimizer.apply_gradients(
            zip(generator_gradients, self.G.trainable_variables)
        )

        encoder_gradients = enc_tape.gradient(total_E_loss, self.E.trainable_variables)
        self.E_optimizer.apply_gradients(
            zip(encoder_gradients, self.E.trainable_variables)
        )
        return (
            total_G_loss,
            total_E_loss,
            gan_loss,
            vaegan_loss,
            l1_loss,
            kl_loss,
            clr_loss,
        )

    @tf.function
    def batch_train(self, batch, current_step, disc_std):
        """Batch train the model.

        Args:
            batch (tensor): tensor with batches of real input and target.
            current_step (tensor): tensor with int value depicting current step.
            disc_std (tensor): tensor with float value depicting current std for disc noise.
        """
        total_D_loss, D_true_loss, D_random_loss, D_encoded_loss = self.train_D(
            batch, disc_std
        )
        (
            total_G_loss,
            total_E_loss,
            gan_loss,
            vaegan_loss,
            l1_loss,
            kl_loss,
            clr_loss,
        ) = self.train_G_E(batch, disc_std)

        with self.summary_writer.as_default():
            tf.summary.scalar("model/total_D_loss", total_D_loss, step=current_step)
            tf.summary.scalar("model/total_G_loss", total_G_loss, step=current_step)
            tf.summary.scalar("model/total_E_loss", total_E_loss, step=current_step)
            tf.summary.scalar("D/D_true_loss", D_true_loss, step=current_step)
            tf.summary.scalar("D/D_random_loss", D_random_loss, step=current_step)
            tf.summary.scalar("D/D_encoded_loss", D_encoded_loss, step=current_step)
            tf.summary.scalar("G_E/gan_loss", gan_loss, step=current_step)
            tf.summary.scalar("G_E/vaegan_loss", vaegan_loss, step=current_step)
            tf.summary.scalar("G_E/l1_loss", l1_loss, step=current_step)
            tf.summary.scalar("G_E/kl_loss", kl_loss, step=current_step)
            tf.summary.scalar("G/clr_loss", clr_loss, step=current_step)
            tf.summary.scalar(
                "model_info/std_D_noise", disc_std, step=current_step,
            )
            tf.summary.scalar(
                "model_info/G_lr", self.G_optimizer.learning_rate, step=current_step
            )
            tf.summary.scalar(
                "model_info/D_lr", self.D_optimizer.learning_rate, step=current_step
            )
            tf.summary.scalar(
                "model_info/E_lr", self.E_optimizer.learning_rate, step=current_step
            )

    def load_models(self, models_directory, version=None):
        """Load all models.

        Args:
            models_directory (Path): Root directory of models.
            version (int): version number.
        """
        version = "_" + str(version) if version else ""
        self.D = load_model(models_directory / f"discriminator{version}.h5")
        self.G = load_model(models_directory / f"generator{version}.h5")
        self.E = load_model(models_directory / f"encoder{version}.h5")

    def save_models(self, models_directory, version=None):
        """Save the model weights.

        Args:
            models_directory (Path): Root directory of models.
            version (int): version number. For these models it's the current step number.
        """
        models_directory.mkdir(parents=True, exist_ok=True)
        version = "_" + str(version) if version else ""

        self.D.save(models_directory / f"discriminator{version}.h5")
        self.G.save(models_directory / f"generator{version}.h5")
        self.E.save(models_directory / f"encoder{version}.h5")

    def create_example(self, example):
        """Creates and stores four examples based on a random input image.

        Args:
            example (tf batch): tensorflow batch.
        """
        for input_image, output_image in example:
            predictions = {}
            for i in np.arange(4):
                z_random = create_z_random(
                    mean=0,
                    std=1,
                    batch_size=self.batch_size,
                    latent_vector=self.latent_vector,
                )
                predicted_image = self.G.predict([input_image.numpy(), z_random])
                predicted_image = (predicted_image[0] * 0.5) + 0.5
                predictions[i] = predicted_image

            fig, axes = plt.subplots(figsize=(15, 3 * 6), nrows=2, ncols=3,)

            input_results = [
                (input_image[0].numpy() * 0.5) + 0.5,
                (output_image[0] * 0.5) + 0.5,
            ] + list(predictions.values())

            titles = ["Input Image", "Ground Truth"] + [
                f"Prediction_{pred_num}" for pred_num in list(predictions.keys())
            ]

            for title, img, ax in zip(titles, input_results, axes.flatten()):
                plt.subplot(ax)
                plt.title(title, fontweight="bold")
                plt.imshow(img)
                plt.axis("off")

            for ax in axes.flatten():
                plt.subplot(ax)
                plt.axis("off")

            plt.tight_layout()
            plt.savefig(str(self.image_storage / f"example_steps_{self.steps}"))
            plt.close()
