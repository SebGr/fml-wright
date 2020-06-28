import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    AveragePooling2D,
    Conv2D,
    LeakyReLU,
    Flatten,
    Dense,
)
from tensorflow.keras.models import Model

from fmlwright.modeling.neural_networks import ResidualBlock


def create_residual_network(
    input_shape,
    normalization,
    activation,
    max_filter_size,
    filter_size,
    n_res_blocks,
    latent_vector,
):
    """Create the residual network for the encoder.

    Args:
        input_shape (int, int, int): 3 dimension input shape.
        normalization (str): Normalization layer to use.
        activation (str): Activation layer to use.
        max_filter_size (int): Maximum size of the filters.
        filter_size (int): Size of first level filter.
        n_res_blocks (int): Number of residual blocks to make. Equals n - 1.
        latent_vector (int): size of z, latent vector.

    Returns:
        Keras model.
    """
    img_input = Input(input_shape, name="enc_input_b")
    init_kernel = tf.random_normal_initializer(0.0, 0.02)
    ResBlock = ResidualBlock(
        n_kernels=3,
        n_strides=1,
        normalization=normalization,
        activation=activation,
        max_filter_size=max_filter_size,
    )

    block = Conv2D(
        filters=filter_size,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=True,
        kernel_initializer=init_kernel,
    )(img_input)

    for i in range(1, n_res_blocks):
        block_filter_size = filter_size * i
        cur_filter = min(block_filter_size, max_filter_size)
        block = ResBlock.build(block, n_filters=cur_filter)

    block = LeakyReLU(0.2)(block)
    block = AveragePooling2D(pool_size=8, strides=8)(block)
    block = Flatten()(block)

    z_mean = Dense(latent_vector, name="z_mean", kernel_initializer=init_kernel)(block)
    z_log_sigma = Dense(
        latent_vector, name="z_log_sigma", kernel_initializer=init_kernel,
    )(block)

    epsilon = tf.random.normal(shape=[latent_vector])
    z = z_mean + tf.exp(z_log_sigma) * epsilon

    model = Model(inputs=img_input, outputs=[z, z_mean, z_log_sigma], name="encoder")
    return model
