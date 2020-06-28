import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, LineString


def create_exterior_points(gdf, number_points_along_line=10):
    """Create an exterior ring around all of the polygons within the geodataframe.

    Every line will get number_points along line of points.

    Args:
        gdf (gpd.GeoDataFrame): Input geodataframe.
        number_points_along_line (int): Number of points to create along a line.

    Returns:
        List[List[int, int]], Where the list items are points.
    """
    locs = []
    for x in gdf.geometry.values:
        if isinstance(x, MultiPolygon):
            x = list(x)
        else:
            x = [x]
        for y in x:
            coords = get_xy_coords(y)
            for i in np.arange(len(coords) - 1):
                locs += find_points_along_line(
                    coords[i], coords[i + 1], number_of_points=number_points_along_line
                )
            locs += coords

    # Keep only unique locations
    unique_locs = []
    for elem in locs:
        if elem not in unique_locs:
            unique_locs.append(elem)
    return unique_locs


def cut_line_at_points(line, points):
    """Cut  linestring at pre defined points.

    Args:
        line (LineString): Original linestring.
        points (list): List of shapely Point.

    Returns:
        List with line subsections.
    """
    # First coords of line
    coords = list(line.coords)

    # Keep list coords where to cut (cuts = 1)
    cuts = [0] * len(coords)
    cuts[0] = 1
    cuts[-1] = 1

    # Add the coords from the points
    coords += [list(p.coords)[0] for p in points]
    cuts += [1] * len(points)

    # Calculate the distance along the line for each point
    dists = [line.project(Point(p)) for p in coords]

    # sort the coords/cuts based on the distances
    coords = [p for (d, p) in sorted(zip(dists, coords))]
    cuts = [p for (d, p) in sorted(zip(dists, cuts))]

    # generate the Lines
    lines = []

    for i in range(len(coords) - 1):
        if cuts[i] == 1:
            # find next element in cuts == 1 starting from index i + 1
            j = cuts.index(1, i + 1)
            lines.append(LineString(coords[i : j + 1]))

    return lines


def find_intersecting_points(gdf_1, gdf_2):
    """Find the intersecting points between two gdf.

    Args:
        gdf_1 (gpd.GeoDataFrame): First geodataframe.
        gdf_2 (gpd.GeoDataFrame): Second geodataframe

    Returns:
        geodataframe with intersecting points.
    """
    points = gdf_1.unary_union.intersection(gdf_2.unary_union)
    if isinstance(points, Point):
        return gpd.GeoDataFrame(geometry=[points])
    else:
        return gpd.GeoDataFrame(geometry=list(points))


def find_points_along_line(point_1, point_2, number_of_points=2):
    """For two points, find points along that line.

    This is very ugly atm and needs improvement.

    Args:
        point_1 (tuple): First point.
        point_2 (tuple): Second point.
        number_of_points (int): Number of points to find.

    Returns:
        list of points along a line, excluding the original points.
    """
    x_dist = point_1[0] - point_2[0]
    y_dist = point_1[1] - point_2[1]

    if (x_dist < 0) and (y_dist >= 0):
        x_dist *= -1
        y_dist *= -1

    elif (x_dist < 0) and (y_dist <= 0):
        x_dist *= -1
        y_dist *= -1

    elif (x_dist >= 0) and (y_dist > 0):
        x_dist *= -1
        y_dist *= -1
    elif (x_dist >= 0) and (y_dist <= 0):
        y_dist *= -1
        x_dist *= -1

    x_step = x_dist / (number_of_points + 1)
    y_step = y_dist / (number_of_points + 1)
    all_points = []
    point = [point_1[0], point_1[1]]
    for _ in np.arange(number_of_points):
        point = point.copy()
        point[0] += x_step
        point[1] += y_step
        all_points.append(point)
    return all_points


def get_angle_between_points(point_a, point_b):
    """Calculate the angle between two points within 360 degrees.

    Args:
        point_a (tuple): First point.
        point_b (tuple): Second point.

    Returns:
        float value between 0 and 360.
    """
    diff_x = point_b[0] - point_a[0]
    diff_y = point_b[1] - point_a[1]
    angle = np.arctan2(diff_y, diff_x)
    return np.degrees(angle) % 360.0


def get_xy_coords(geom):
    """For a shapely shape, return the exterior coordinates.

    Args:
        geom: Shapely shape.

    Returns:
        List of tuples for exterior coordinates.
    """
    return list(geom.exterior.coords)


def transform_contour_to_polygon(contour):
    """Transform a contour to a polygon.

    Args:
        contour (np.array): original contours

    Returns:
        Polygon or none depending on number of contour points.
    """
    x, _, z = contour.shape
    area = list(contour.reshape((x, z)))
    if len(area) <= 1:
        return None
    else:
        area.append(area[-1])
        return Polygon(area)
