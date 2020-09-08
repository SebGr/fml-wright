from fmlwright.trainer.neural_networks import PatchBlock
from tensorflow.keras.layers import Input, ZeroPadding2D, Conv2D
from tensorflow.keras.models import Model

import tensorflow as tf


def create_patchgan(
    filter_size, dropout, input_shape, name, activation, normalization, num_D
):
    """Create a patchgan model.

    Args:
        filter_size (int): Size of first level filter.
        dropout (float): Include dropout for intermediate layers.
        input_shape (int, int, int): 3 dimension input shape.
        name (str): Name of network.
        activation (str): Activation layer to use.
        normalization (str): Normalization layer to use.

    Returns:
        A keras PatchGan model.
    """
    input_b = Input(input_shape, name=f"{name}_input_b")

    PatchGANBlock = PatchBlock(
        n_kernels=4,
        n_strides=2,
        dropout=dropout,
        normalization=normalization,
        activation=activation,
    )

    def _build_model(PatchGANBlock, filter_size, input_tensor):
        conv_block = PatchGANBlock.build(
            input_tensor, n_filters=filter_size * 1, norm_layer=False
        )
        conv_block = PatchGANBlock.build(conv_block, n_filters=filter_size * 2)
        conv_block = PatchGANBlock.build(conv_block, n_filters=filter_size * 4)
        conv_block = ZeroPadding2D()(conv_block)
        conv_block = PatchGANBlock.build(
            conv_block, n_strides=1, n_filters=filter_size * 8, padding=False
        )
        conv_block = ZeroPadding2D()(conv_block)

        conv_block = Conv2D(
            filters=1,
            kernel_size=4,
            strides=1,
            kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
            padding="valid",
            use_bias=True,
        )(conv_block)

        return conv_block

    output = _build_model(PatchGANBlock, filter_size, input_b)

    return Model(inputs=input_b, outputs=output, name=name)
