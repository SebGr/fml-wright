import numpy as np


def create_z_random(latent_vector, mean=0, std=1, batch_size=1):
    """Create the z_random discribution.

    Args:
        latent_vector (int): Latent dimension value.
        mean (int): Mean value.
        std (int): Standard deviation.
        batch_size (int): Batch size.

    Returns:
        tensor with size of batch_size and latent vector, filled with random values.
    """
    noise = np.random.normal(mean, std, size=[batch_size, latent_vector]).astype(
        np.float32
    )
    return noise
