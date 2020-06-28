import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Conv2D,
    Dropout,
    LeakyReLU,
    RepeatVector,
    Reshape,
    ReLU,
    BatchNormalization,
    LayerNormalization,
    AveragePooling2D,
    Conv2DTranspose,
)
from tensorflow_addons.layers import InstanceNormalization

import logging

log = logging.getLogger(__name__)


class LatentNoiseBlock:
    """Noise block to add noise to an input vector."""

    def __init__(self, latent_vector):
        """Initialize the noise block.

        Args:
            latent_vector (int): size of z, latent vector.
        """
        self.latent_vector = latent_vector

    def build(self, input_tensor, cur_z_shape):
        """Build a Latent noise block.

        Args:
            input_tensor (tf.tensor): Input tensor.
            cur_z_shape (int): current x, y size of network.

        Returns:
            Tensor for a Latent noise block.
        """
        cur_z_shape = int(cur_z_shape)
        repeater = RepeatVector(int(cur_z_shape * cur_z_shape), dtype=tf.float32)(
            input_tensor
        )
        single_noise_filter = Reshape([cur_z_shape, cur_z_shape, self.latent_vector])(
            repeater
        )
        return single_noise_filter


def activation_layer(activation_type):
    """Activation layer to use.

    Args:
        activation_type (str): Activation type to use.

    Returns:
        Tensor for the chosen activation layer.
    """
    if activation_type == "leakyrelu":
        activation = LeakyReLU(0.2)
    elif activation_type == "relu":
        activation = ReLU()
    else:
        log.error("Unknown activation type, using a ReLU.")
        activation = ReLU()
    return activation


def normalization_layer(normalization_type):
    """Normalization layer to use.

    Args:
        normalization_type (str): Normalization type to use.

    Returns:
        Tensor for the chosen normalization layer.
    """
    if normalization_type == "instance":
        normalization = InstanceNormalization(
            axis=-1, center=False, scale=False, epsilon=1e-05
        )
    elif normalization_type == "batch":
        normalization = BatchNormalization(epsilon=1e-05, momentum=0.9)
    elif normalization_type == "layer":
        normalization = LayerNormalization()
    else:
        log.error("Unknown normalization type, using batchnorm.")
        normalization = BatchNormalization()
    return normalization


class DownConvBlock:
    """Class for the Down Convolution block for Unet."""

    def __init__(self, n_kernels, n_strides, activation, normalization):
        """Initialize the Down Convolution Block.

        Args:
            n_kernels (int): Number of kernels for Conv2D.
            n_strides (int): Stride size.
            activation (str): Type of activation layer to use.
            normalization (str): Type of normalization layer to use.
        """
        self.n_kernels = n_kernels
        self.n_strides = n_strides
        self.norm_type = normalization
        self.activation_type = activation
        self.kernel_init = tf.random_normal_initializer(0.0, 0.02)

    def build(
        self,
        input_tensor,
        n_filters,
        use_activation=True,
        use_normalization=True,
        use_bias=False,
    ):
        """Build a DownConv block.

        Args:
            input_tensor (tf.tensor): Input tensor.
            n_filters (int): Number of filters.
            use_activation (bool): Whether to use activation layer.
            use_normalization (bool): Whether to use normalization layer.
            use_bias (bool): Whether to use bias.

        Returns:
            A tensor with multiple layers for the built block.
        """
        activation = Conv2D(
            filters=n_filters,
            kernel_size=self.n_kernels,
            strides=self.n_strides,
            padding="same",
            kernel_initializer=self.kernel_init,
            use_bias=use_bias,
        )(input_tensor)

        if use_normalization:
            activation = normalization_layer(self.norm_type)(activation)

        if use_activation:
            activation = activation_layer(self.activation_type)(activation)

        return activation


class UpConvBlock:
    """Class for the Up Convolution block for Unet."""

    def __init__(self, n_kernels, n_strides, dropout, activation, normalization):
        """Initialize the Up Convolution Block.

        Args:
            n_kernels (int): Number of kernels for Conv2D.
            n_strides (int): Stride size.
            dropout (float): Include dropout for intermediate layers.
            activation (str): Type of activation layer to use.
            normalization (str): Type of normalization layer to use.
        """
        self.n_kernels = n_kernels
        self.n_strides = n_strides
        self.dropout = dropout
        self.norm_type = normalization
        self.activation_type = activation
        self.init_kernel = tf.random_normal_initializer(0.0, 0.02)

    def build(
        self,
        input_tensor,
        conv_block_tensor,
        n_filters,
        use_dropout=False,
        use_normalization=True,
        use_activation=True,
        use_bias=False,
    ):
        """Build an up convolution block.

        Args:
            input_tensor (tf.tensor): Input tensor.
            conv_block_tensor (tf.tensor): Down conv block tensor.
            n_filters (int): Number of filters.
            use_dropout (bool): Whether to use dropout.
            use_normalization (bool): Whether to use normalization layer.
            use_activation (bool): Whether to use activation layer.
            use_bias (bool): Whether to use bias.

        Returns:
            A tensor with multiple layers for the up convolution block.
        """
        activation = Conv2DTranspose(
            n_filters,
            kernel_size=self.n_kernels,
            strides=self.n_strides,
            padding="same",
            kernel_initializer=self.init_kernel,
            use_bias=use_bias,
        )(input_tensor)

        if use_normalization:
            activation = normalization_layer(self.norm_type)(activation)

        if use_activation:
            activation = activation_layer(self.activation_type)(activation)

        if use_dropout:
            activation = Dropout(rate=self.dropout)(activation)

        concat_layer = layers.concatenate([activation, conv_block_tensor], axis=3)
        return concat_layer


class ResidualBlock:
    """Class to build Residual blocks for the Encoder."""

    def __init__(
        self,
        n_kernels,
        n_strides,
        activation,
        normalization,
        max_filter_size,
        filter_step=64,
    ):
        """Initialize the Residuel Blocks.

        Args:
            n_kernels (int): Number of kernels for Conv2D.
            n_strides (int): Stride size.
            activation (str): Type of activation layer to use.
            normalization (str): Type of normalization layer to use.
            max_filter_size (int): Maximum filter size.
            filter_step (int): Step size for the filters.
        """
        self.n_kernels = n_kernels  # 3
        self.n_strides = n_strides  # 1
        self.norm_type = normalization
        self.activation_type = activation
        self.filter_step = filter_step
        self.max_filter_size = max_filter_size

        self.init_kernel = tf.random_normal_initializer(0.0, 0.02)

    def build(
        self, input_tensor, n_filters,
    ):
        """Build an encoder block.

        Args:
            input_tensor (tf.tensor): Input tensor.
            n_filters (int): Number of filters.

        Returns:
            Tensor with multiple layers for the Residual block.
        """
        # Block 1
        activation = normalization_layer(self.norm_type)(input_tensor)
        activation = activation_layer(self.activation_type)(activation)
        activation = Conv2D(
            n_filters,
            kernel_size=self.n_kernels,
            strides=self.n_strides,
            padding="same",
            kernel_initializer=self.init_kernel,
            use_bias=True,
        )(activation)

        activation = normalization_layer(self.norm_type)(activation)
        activation = activation_layer(self.activation_type)(activation)

        next_filter_size = min(self.max_filter_size, n_filters + self.filter_step)
        activation = Conv2D(
            next_filter_size,
            kernel_size=self.n_kernels,
            strides=self.n_strides,
            padding="same",
            kernel_initializer=self.init_kernel,
            use_bias=True,
        )(activation)
        activation = AveragePooling2D(pool_size=(2, 2), strides=2)(activation)

        skip_block = AveragePooling2D(pool_size=(2, 2), strides=2)(input_tensor)
        skip_block = Conv2D(next_filter_size, 1, 1, use_bias=False)(skip_block)
        resblock = layers.add([activation, skip_block])
        return resblock


class PatchBlock:
    """Class to create Patch blocks for the Discriminator."""

    def __init__(self, n_kernels, n_strides, dropout, activation, normalization):
        """Initialize the Patchblock.

        Args:
            n_kernels (int): Number of kernels for Conv2D.
            n_strides (int): Stride size.
            dropout (float): Include dropout for intermediate layers.
            activation (str): Type of activation layer to use.
            normalization (str): Type of normalization layer to use.
        """
        self.n_kernels = n_kernels
        self.n_strides = n_strides
        self.dropout = dropout
        self.norm_type = normalization
        self.activation_type = activation

    def build(
        self,
        input_tensor,
        n_filters,
        n_kernels=None,
        n_strides=None,
        padding=True,
        activation_type=True,
        norm_layer=True,
    ):
        """Build an initialized Patch block.

        Args:
            input_tensor (tf.tensor): Input tensor.
            n_filters (int): Number of filters.
            n_kernels (int): Number of kernels for Conv2D.
            n_strides (int): Stride size.
            padding (bool): Use padding='same' or no padding.
            activation_type (bool): Whether to use activation layer.
            norm_layer (bool): Whether to use normalization layer.

        Returns:
            Tensor with multiple layers for the Patch block.
        """
        if padding:
            activation = Conv2D(
                filters=n_filters,
                kernel_size=n_kernels if n_kernels else self.n_kernels,
                strides=n_strides if n_strides else self.n_strides,
                padding="same",
                kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
                use_bias=True,
            )(input_tensor)
        else:
            activation = Conv2D(
                filters=n_filters,
                kernel_size=n_kernels if n_kernels else self.n_kernels,
                strides=n_strides if n_strides else self.n_strides,
                kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
                padding="valid",
                use_bias=True,
            )(input_tensor)

        if norm_layer:
            activation = normalization_layer(self.norm_type)(activation)

        if self.dropout > 0:
            activation = Dropout(rate=self.dropout)(activation)

        if activation_type:
            activation_l = activation_layer(self.activation_type)
            activation = activation_l(activation)

        return activation
