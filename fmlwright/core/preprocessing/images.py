import logging

import cv2
import geopandas as gpd
import numpy as np

from fmlwright.core.labeling import get_room_color_map
from fmlwright.core.utils import transform_contour_to_polygon

log = logging.getLogger(__name__)


def resize_and_pad(img, size, pad_color=0):
    """Resize and pad image.

    Args:
        img (np.array): image.
        size (tuple): size to resize to.
        pad_color (int): pad color.

    Returns:
        Resized and padded image.
    """
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w / h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1:  # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = (
            np.floor(pad_horz).astype(int),
            np.ceil(pad_horz).astype(int),
        )
        pad_top, pad_bot = 0, 0
    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(
        pad_color, (list, tuple, np.ndarray)
    ):  # color image but only one color provided
        pad_color = [pad_color] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(
        scaled_img,
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color,
    )

    return scaled_img


def image_to_geodataframe(image):
    """Turn an image into a geodataframe.

    Args:
        image (np.array): opencv image.

    Returns:
        Geodataframe of the image.
    """
    colormap = get_room_color_map()
    all_contours = []
    category = []

    for key, color in colormap.items():
        mask = cv2.inRange(
            src=image, lowerb=np.array(color) - 20, upperb=np.array(color) + 20
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours += contours
        category += [key] * len(contours)

    poly_list = []
    categories = []
    for _contour, _cat in zip(all_contours, category):
        if len(_contour) > 2:
            categories.append(_cat)
            poly_list.append(transform_contour_to_polygon(_contour))

    gdf = gpd.GeoDataFrame(geometry=poly_list)
    gdf["category"] = categories
    gdf["colors"] = (
        gdf["category"].apply(lambda cat: [col / 255 for col in colormap[cat]]).values
    )
    gdf.geometry = gdf.scale(yfact=-1, origin=(0, 0))
    return gdf


def image_to_walls(img):
    """From an image, retrieve the walls.

    Aimed at usage for the structure input images.

    Args:
        img (np.array): Input image from structure dataset.

    Returns:
        geodataframe with the walls as a single polygon.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(gray_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    poly_list = [transform_contour_to_polygon(_contour) for _contour in contours]
    poly_list = [x for x in poly_list if x]

    rooms = gpd.GeoDataFrame(geometry=poly_list)

    rooms["area"] = rooms.area
    rooms = rooms.sort_values("area", ascending=False)
    rooms = rooms.tail(-1).reset_index(
        drop=True
    )  # drop the largest area area outside of the floor

    # find floors:
    floors = (
        gpd.GeoDataFrame(geometry=[rooms.unary_union]).explode().reset_index(drop=True)
    )

    # find best overlapping areas so they can be removed.
    best_matches = {}
    for i, _floor in floors.iterrows():
        _floor = _floor.geometry
        best_overlap = 0
        for j, _room in rooms.iterrows():
            _room = _room.geometry
            if _floor.intersects(_room):
                overlap = _floor.intersection(_floor).area / _floor.area * 100
                if overlap > best_overlap:
                    best_matches[i] = j
                    best_overlap = overlap
    rooms = rooms.drop(best_matches.values(), axis=0)

    res_union = gpd.overlay(floors, rooms, how="difference")

    res_union.geometry = res_union.scale(yfact=-1, origin=(0, 0))
    return res_union
