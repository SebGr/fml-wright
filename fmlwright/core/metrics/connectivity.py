import networkx as nx


def find_neighbouring_areas(gdf):
    """Find neighbouring areas.

    Args:
        gdf (gpd.GeoDataFrame): Geodataframe of the building floorplan.

    Returns:
        Dictionary with rooms and neighbouring room ids.
    """
    buffer = 2  # add small buffer to ensure overlap.

    matching_ids = {}
    for i, row in gdf.iterrows():
        matching_id = gdf[~gdf.geometry.buffer(buffer).disjoint(row.geometry)].index
        matching_id = [x for x in matching_id if i != x]
        matching_ids[i] = matching_id

    return matching_ids


def create_graph(matching_ids, gdf):
    """Create a networkx graph for the neigbouring areas.

    Args:
        matching_ids (dict): Dictionary with neighbouring areas.
        gdf (gpd.GeoDataFrame): Geodataframe of the building floorplan.

    Returns:
        nx.Graph with floorplan connectivity.
    """
    G = nx.Graph()
    edges = []
    for key, val in matching_ids.items():
        for _v in val:
            edges.append((key, _v))

    pos = {}
    for i, row in gdf.iterrows():
        pos[i] = [row.geometry.centroid.x, row.geometry.centroid.y]

    G.add_edges_from(edges)
    for x in G.nodes():
        coords = pos[x]
        G.nodes[x]["x"] = coords[0]
        G.nodes[x]["y"] = coords[1]
    return G


def get_coords(G):
    """Get the x, y coordinates of a graph.

    Args:
        G (nx.Graph): networkx graph.

    Returns:
        Dictionary with coordinates.
    """
    coords = {}
    for x in G.nodes(data=True):
        coords[x[0]] = [x[1]["x"], x[1]["y"]]
    return coords
