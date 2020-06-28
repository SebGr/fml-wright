import numpy as np
import pandas as pd


def calculate_angle_to_rotate(top_x, top_y, low_x, low_y):
    """Calculate angle between two points in order to know how to rotate.

    Args:
        top_x (int): x coordinate of higher point.
        top_y (int): y coordinate of higher point.
        low_x (int): x coordinate of lower point.
        low_y (int): y coordinate of lower point.

    Returns:
        Angle to rotate in degrees.
    """
    delta_x = np.abs(top_x - low_x)
    delta_y = np.abs(top_y - low_y)

    m = delta_x / delta_y

    angle = np.arctan(m)
    angle = np.rad2deg(angle)

    if top_y > low_y:
        angle *= -1
    return angle


def find_angle(polygon):
    """Find the angle of a polygon to rotate.

    Args:
        polygon (Polygon): Shapely polygon of a building floorplan.

    Returns:
        Angle to rotate in degrees.
    """
    coordinates = list(polygon.simplify(0.01).values[0].exterior.coords)
    rightmost_coords = list(
        pd.DataFrame(coordinates).sort_values([0]).drop_duplicates().tail(2).values
    )
    angle = calculate_angle_to_rotate(
        rightmost_coords[0][0],
        rightmost_coords[0][1],
        rightmost_coords[1][0],
        rightmost_coords[1][1],
    )
    return angle
