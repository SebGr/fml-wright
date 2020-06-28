def get_room_label_map():
    """Get the label mapping number."""
    labelmap = dict()
    labelmap["living_room"] = 1
    labelmap["kitchen"] = 2
    labelmap["bedroom"] = 3
    labelmap["bathroom"] = 4
    labelmap["restroom"] = 4
    labelmap["washing_room"] = 4
    labelmap["office"] = 3
    labelmap["closet"] = 6
    labelmap["balcony"] = 7
    labelmap["corridor"] = 8
    labelmap["dining_room"] = 1
    labelmap["laundry_room"] = 6
    labelmap["PS"] = 9
    return labelmap


def get_room_color_map():
    """Get the label mapping color."""
    colormap = dict()
    colormap["living_room"] = [255, 51, 51]
    colormap["kitchen"] = [32, 178, 170]
    colormap["bedroom"] = [0, 255, 127]
    colormap["bathroom"] = [255, 0, 255]
    colormap["restroom"] = [255, 0, 255]
    colormap["washing_room"] = [255, 160, 122]
    colormap["office"] = [0, 255, 127]
    colormap["closet"] = [255, 160, 122]
    colormap["balcony"] = [255, 255, 0]
    colormap["corridor"] = [65, 105, 225]
    colormap["dining_room"] = [255, 51, 51]
    colormap["laundry_room"] = [255, 160, 122]
    colormap["PS"] = [80, 128, 255]
    colormap["entrance"] = [255, 80, 128]
    colormap["stairs"] = [80, 255, 128]
    colormap["unknown"] = [255, 230, 180]
    colormap["wall"] = [0, 0, 0]
    colormap["door"] = [0, 0, 0]
    return colormap
