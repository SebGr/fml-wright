import logging

import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model

from fmlwright.trainer.neural_networks import (
    DownConvBlock,
    UpConvBlock,
    LatentNoiseBlock,
)

log = logging.getLogger(__name__)


def create_unet(
    input_shape,
    latent_vector,
    filter_size,
    dropout,
    activation,
    normalization,
    z_input_layer,
):
    """Create the unet bicycle model.

    Args:
        input_shape (int, int, int): 3 dimension input shape.
        latent_vector (int): size of z, latent vector.
        filter_size (int): Size of first level filter.
        dropout (float): Include dropout for intermediate layers.
        activation (str): Activation layer to use.
        normalization (str): Normalization layer to use.
        z_input_layer (str): Where to add z. options are ['first', 'intermediate'].

    Returns:
        A keras Unet model for BicycleGAN.
    """
    input_image = Input(shape=input_shape, name="gen_input_a")
    input_noise = Input(shape=[latent_vector], name="gen_input_z")
    z_shape_level = input_shape[0]

    DownBlock = DownConvBlock(
        n_kernels=4, n_strides=2, activation=activation, normalization=normalization,
    )

    UpBlock = UpConvBlock(
        n_kernels=4,
        n_strides=2,
        dropout=dropout,
        activation="relu",
        normalization=normalization,
    )

    LatBlock = LatentNoiseBlock(latent_vector=latent_vector)

    if latent_vector != 0:
        # input
        input_block = concatenate(
            [input_image, LatBlock.build(input_noise, cur_z_shape=z_shape_level)]
        )

        # layer 1
        conv_block_1 = DownBlock.build(
            input_tensor=input_block, n_filters=filter_size, use_normalization=False
        )
        conv_block_noise_1 = concatenate(
            [conv_block_1, LatBlock.build(input_noise, cur_z_shape=z_shape_level / 2)]
        )
    else:
        conv_block_1 = DownBlock.build(
            input_tensor=input_image, n_filters=filter_size, use_normalization=False
        )
        if z_input_layer != "first":
            log.error(
                "latent vector cannot be 0 at the same time as z_input_layer being "
                "intermediate."
            )
            exit()

    if z_input_layer == "intermediate":
        # layer 2
        conv_block_2 = DownBlock.build(
            input_tensor=conv_block_noise_1, n_filters=filter_size * 2
        )
        conv_block_noise_2 = concatenate(
            [conv_block_2, LatBlock.build(input_noise, cur_z_shape=z_shape_level / 4)]
        )

        # layer 3
        conv_block_3 = DownBlock.build(
            input_tensor=conv_block_noise_2, n_filters=filter_size * 4
        )
        conv_block_noise_3 = concatenate(
            [conv_block_3, LatBlock.build(input_noise, cur_z_shape=z_shape_level / 8)]
        )
        # layer 4
        conv_block_4 = DownBlock.build(
            input_tensor=conv_block_noise_3, n_filters=filter_size * 8
        )
        conv_block_noise_4 = concatenate(
            [conv_block_4, LatBlock.build(input_noise, cur_z_shape=z_shape_level / 16)]
        )

        # layer 5
        conv_block_5 = DownBlock.build(
            input_tensor=conv_block_noise_4, n_filters=filter_size * 8
        )
        conv_block_noise_5 = concatenate(
            [conv_block_5, LatBlock.build(input_noise, cur_z_shape=z_shape_level / 32)]
        )

        # layer 6
        conv_block_6 = DownBlock.build(
            input_tensor=conv_block_noise_5, n_filters=filter_size * 8
        )
        conv_block_noise_6 = concatenate(
            [conv_block_6, LatBlock.build(input_noise, cur_z_shape=z_shape_level / 64)]
        )
        # layer 7
        conv_block_7 = DownBlock.build(
            input_tensor=conv_block_noise_6, n_filters=filter_size * 8
        )
        conv_block_noise_7 = concatenate(
            [conv_block_7, LatBlock.build(input_noise, cur_z_shape=z_shape_level / 128)]
        )

        # center
        conv_center = DownBlock.build(
            input_tensor=conv_block_noise_7,
            n_filters=filter_size * 8,
            use_normalization=False,
        )
    elif z_input_layer == "first":
        conv_block_2 = DownBlock.build(
            input_tensor=conv_block_1, n_filters=filter_size * 2
        )
        conv_block_3 = DownBlock.build(
            input_tensor=conv_block_2, n_filters=filter_size * 4
        )
        conv_block_4 = DownBlock.build(
            input_tensor=conv_block_3, n_filters=filter_size * 8
        )
        conv_block_5 = DownBlock.build(
            input_tensor=conv_block_4, n_filters=filter_size * 8
        )
        conv_block_6 = DownBlock.build(
            input_tensor=conv_block_5, n_filters=filter_size * 8
        )
        conv_block_7 = DownBlock.build(
            input_tensor=conv_block_6, n_filters=filter_size * 8
        )

        # center
        conv_center = DownBlock.build(
            input_tensor=conv_block_7, n_filters=filter_size * 8,
        )
    else:
        raise ValueError("Unknown z input layer, quitting")

    deconv_block_1 = UpBlock.build(conv_center, conv_block_7, n_filters=filter_size * 8)
    deconv_block_2 = UpBlock.build(
        deconv_block_1, conv_block_6, n_filters=filter_size * 8, use_dropout=True
    )
    deconv_block_3 = UpBlock.build(
        deconv_block_2, conv_block_5, n_filters=filter_size * 8, use_dropout=True
    )
    deconv_block_4 = UpBlock.build(
        deconv_block_3, conv_block_4, n_filters=filter_size * 8, use_dropout=True
    )
    deconv_block_5 = UpBlock.build(
        deconv_block_4, conv_block_3, n_filters=filter_size * 4
    )
    deconv_block_6 = UpBlock.build(
        deconv_block_5, conv_block_2, n_filters=filter_size * 2
    )
    deconv_block_7 = UpBlock.build(
        deconv_block_6, conv_block_1, n_filters=filter_size * 1
    )

    # Final upsample
    model_output = Conv2DTranspose(
        3,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="tanh",
        kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
    )(deconv_block_7)

    return Model(
        inputs=[input_image, input_noise], outputs=model_output, name="generator"
    )
