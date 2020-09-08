import logging

import numpy as np

from fmlwright.predictor.Pix2Pix import BasePredictor

log = logging.getLogger(__name__)


class Pix2PixPredictor(BasePredictor):
    """Predictor class for the Pix2Pix models."""

    def __init__(self, model_location, categories, conf_location):
        """Initialize the predictor with a model.

        Args:
            model_location (str): path to the root directory for the models.
            categories (list): List of categories.
            conf_location (str): path to config location.
        """
        super().__init__(model_location, categories, conf_location)
        self.model_type = "Pix2Pix"

        self.load_configuration()
        self.load_model()

    def predict(self, img, n_samples, categories=None):
        """Create a prediction for the generator model.

        The predict will generate n_samples per category. If categories is None, it will generate
            prediction for every known model.

        Args:
            img: Image as it was loaded in using cv2.
            n_samples (int): Number of samples to create.
            categories (list): list of categories to predict.

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
                _pred = self._model[_cat].predict(preprocessed_img)
                _pred = (_pred * 0.5) + 0.5
                predictions[(_cat, i)] = _pred[0]
        return predictions
