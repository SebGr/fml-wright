import logging

from fmlwright.generator.BiCycleGAN import SingleStepGenerator

log = logging.getLogger(__name__)


def run(config):
    """Build the generator class.

    Args:
        config (dict): dictionary that is a result of the config/generator file being loaded.

    Returns:
        Generator interface class.
    """
    generator = SingleStepGenerator(
        model_location=config["settings"]["models_directory"],
        input_shape=config["settings"]["input_shape"],
        categories=config["settings"]["categories"],
    )
    generator.load_model()

    return generator
