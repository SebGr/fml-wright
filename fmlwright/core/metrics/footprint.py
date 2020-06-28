import numpy as np
import pandas as pd
from shapely.geometry import Point

from fmlwright.core.utils import find_points_along_line


def calculate_centroid_distance(geom_shape, number_of_points=0):
    """For a polygon, it will calculate the distance to to the center of the object.

    Args:
        geom_shape (Polygon): should be a polygon.
        number_of_points (int): number of points along a line that should be created.

    Returns:
        pd.DataFrame containing the distances to the centroid, degrees and xy coordinates.
    """
    points = list(geom_shape.boundary.coords)
    locs = []
    if number_of_points != 0:
        for i in np.arange(len(points) - 1):
            locs += [points[i]]
            locs += find_points_along_line(
                points[i], points[i + 1], number_of_points=number_of_points
            )
        locs += [points[-1]]
    else:
        locs = points

    res = []
    [res.append(geom_shape.centroid.distance(Point(_loc))) for _loc in locs]

    temp = pd.DataFrame({"distance": res})
    degrees = 360 / (len(res) - 1)
    temp["degrees"] = temp.index * degrees
    temp["points"] = locs
    return temp
