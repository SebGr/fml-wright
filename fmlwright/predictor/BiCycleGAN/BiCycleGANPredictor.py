import logging

from fmlwright.core import postprocessing
from fmlwright.predictor.BiCycleGAN import BaseBiCycleGAN

log = logging.getLogger(__name__)


class BiCycleGANPredictor(BaseBiCycleGAN):
    """Predictor class for the BiCycleGAN models."""

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
            cleaned_gdf = postprocessing.postprocess_prediction(input_img, _pred)
            processed_results[key] = cleaned_gdf
        return processed_results
