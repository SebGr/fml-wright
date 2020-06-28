from .labels import MODEL_CATEGORIES


def create_floorplan_categories(row):
    """Create the floorplan categories.

    Args:
        row (dict): row with count of rooms available.

    Returns:
        dictionary with categories.
    """
    categories = {}
    for cat in MODEL_CATEGORIES:
        categories[cat] = False

    if "bedroom" in row.keys():
        if row["bedroom"] == 1:
            categories["single_bedroom"] = True
        if row["bedroom"] == 2:
            categories["double_bedroom"] = True
        if row["bedroom"] >= 3:
            categories["multiple_bedroom"] = True
    if "bathroom" in row.keys():
        if row["bathroom"] == 1:
            categories["single_bathroom"] = True
        if row["bathroom"] == 2:
            categories["double_bathroom"] = True
        if row["bathroom"] >= 3:
            categories["multiple_bathroom"] = True
    if "stairs" in row.keys():
        if (row["stairs"] == 2) or (row["stairs"] == 1):
            categories["double_floor"] = True
        if row["stairs"] >= 3:
            categories["multiple_floor"] = True
        else:
            categories["single_floor"] = True
    else:
        categories["single_floor"] = True
    if "balcony" in row.keys():
        if row["balcony"] >= 1:
            categories["balcony"] = True
    if "washing_room" in row.keys():
        if row["washing_room"] >= 1:
            categories["washing_room"] = True
    if "living_room" in row.keys():
        if row["living_room"] >= 1:
            categories["living_room"] = True
    return categories
