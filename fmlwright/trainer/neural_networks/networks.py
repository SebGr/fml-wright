import logging

from fmlwright.trainer.neural_networks.patchgan import create_patchgan
from fmlwright.trainer.neural_networks.residual import create_residual_network
from fmlwright.trainer.neural_networks.unet import create_unet

log = logging.getLogger(__name__)


def create_generator(conf, input_shape, latent_vector):
    """Create the generator model.

    Args:
        conf (dict): Generator part of the config dictionary.
        input_shape (tuple): Input shape.
        latent_vector (int): latent vector.

    Returns:
        Keras model.
    """
    model_type = conf["model_type"]
    neural_network = conf["neural_network"]
    if model_type == "BiCycleGAN":
        if neural_network == "Unet":
            model = create_unet(
                input_shape=input_shape,
                latent_vector=latent_vector,
                filter_size=conf["filter_size"],
                activation=conf["activation"],
                normalization=conf["normalization"],
                dropout=conf["dropout"],
                z_input_layer=conf["z_input_layer"],
            )
        else:
            raise ValueError("Unknown algorithm selected.")
    elif model_type == "Pix2Pix":
        if neural_network == "Unet":
            model = create_unet(
                input_shape=input_shape,
                latent_vector=latent_vector,
                filter_size=conf["filter_size"],
                activation=conf["activation"],
                normalization=conf["normalization"],
                dropout=conf["dropout"],
                z_input_layer="first",
            )
        else:
            raise ValueError("Unknown algorithm selected.")
    else:
        raise ValueError("Unknown model type selected..")
    return model


def create_encoder(conf, input_shape, latent_vector):
    """Create the encoder model.

    Args:
        conf (dict): Encoder part of the config dictionary.
        input_shape (tuple): Input shape.
        latent_vector (int): latent vector.

    Returns:
        Keras model.
    """
    model = create_residual_network(
        input_shape=input_shape,
        normalization=conf["normalization"],
        activation=conf["activation"],
        max_filter_size=conf["max_filter_size"],
        filter_size=conf["filter_size"],
        n_res_blocks=conf["n_res_blocks"],
        latent_vector=latent_vector,
    )
    return model


def create_discriminator(conf, input_shape):
    """Create the discriminator model.

    Args:
        conf (dict): Discriminator part of the config dictionary.
        input_shape (tuple): Input shape.

    Returns:
        Keras model.
    """
    model = create_patchgan(
        input_shape=input_shape,
        filter_size=conf["filter_size"],
        dropout=conf["dropout"],
        activation=conf["activation"],
        normalization=conf["normalization"],
        name="discriminator",
        num_D=conf["num_D"],
    )
    return model
