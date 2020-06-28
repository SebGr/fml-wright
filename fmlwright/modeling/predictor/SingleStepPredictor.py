import logging

import numpy as np
from fmlwright.core import utils, postprocessing
from fmlwright.modeling.predictor import BasePredictor

log = logging.getLogger(__name__)


class SingleStepPredictor(BasePredictor):
    """Predictor class for the BiCycleGAN models."""

    def __init__(self, model_location, categories, conf_location, latent_vector=8):
        """Initialize the predictor with a model.

        Args:
            model_location (str): path to the root directory for the models.
            categories (list): List of categories.
            conf_location (str): path to config location.
            latent_vector (int): Latent dimension value.
        """
        super().__init__(model_location, categories, conf_location)
        self.model_type = "BiCycleGAN"
        self.latent_vector = latent_vector

        self.load_configuration()
        self.load_model()

    def predict(self, img, n_samples, z_random=None, categories=None):
        """Create a prediction for the generator model.

        If z_random is none, a random z will be generated with mean 0 and std 1 per sample. The
        predict will generate n_samples per category. If categories is None, it will generate
        prediction for every known model.

        Args:
            img: Image as it was loaded in using cv2.
            n_samples (int): Number of samples to create.
            z_random (np.array): if a specific z vector needs to be tested.
            categories (list): list of categories to visualize.

        Returns:
            dictionary with the predictions normalized between 0 and 1.
        """
        categories = categories if categories else self.categories
        preprocessed_img = self.preprocess_image(img)
        preprocessed_img = np.reshape(
            preprocessed_img,
            (1, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
        )

        predictions = {}
        for _cat in categories:
            for i in np.arange(n_samples):
                noise = (
                    z_random if z_random else utils.create_z_random(self.latent_vector)
                )
                _pred = self._model[_cat].predict([preprocessed_img, noise])
                _pred = (_pred * 0.5) + 0.5
                predictions[(_cat, i)] = _pred[0]
        return predictions

    def postprocess_predictions(self, predictions, input_img):
        """Postprocess predictions.

        This first does voronoi and then straightening on the floorplan_geodataframe. Finally it
        simplifies the final gdf with the set tolerance.

        Args:
            predictions (dictionary): Dictionary with predictions.
            input_img (np.array): Input image

        Returns:
            Dictionary with processed predictions.
        """
        processed_results = {}
        for key, _pred in predictions.items():
            cleaned_gdf = postprocessing.postprocess_prediction_to_gdf(input_img, _pred)
            processed_results[key] = cleaned_gdf
        return processed_results
