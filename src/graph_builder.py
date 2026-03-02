import networkx as nx
from math import radians, sin, cos, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in km between two lat/lon points.
    Used as edge weight for realistic flight distances."""
    R = 6371.0  # Earth radius km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def build_graph(airports_df, routes_df, selected_airports=None):
    """Build directed NetworkX graph.
    Nodes: airports (IATA) with lat/lon attributes.
    Edges: direct flights with 'distance' weight."""
    G = nx.DiGraph()

    # Filter to selected airports (for performance - 12 major US hubs)
    if selected_airports:
        airports_df = airports_df[airports_df['iata'].isin(selected_airports)].copy()

    # Add nodes
    for _, row in airports_df.iterrows():
        iata = row['iata']
        G.add_node(iata,
                   lat=row['lat'],
                   lon=row['lon'],
                   name=row['name'],
                   city=row['city'])

    # Filter routes to selected airports
    if selected_airports:
        routes_df = routes_df[
            (routes_df['source'].isin(selected_airports)) &
            (routes_df['dest'].isin(selected_airports))
        ]

    # Add edges with distance
    for _, row in routes_df.iterrows():
        src, dst = row['source'], row['dest']
        if src in G and dst in G and src != dst:
            dist = haversine(
                G.nodes[src]['lat'], G.nodes[src]['lon'],
                G.nodes[dst]['lat'], G.nodes[dst]['lon']
            )
            G.add_edge(src, dst, distance=dist, airline=row['airline'])

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

def get_distance_matrix(G, airport_list):
    """Precompute shortest-path distances between all pairs in selected airports.
    Uses Dijkstra on distance weights. Inf if no path."""
    dist_matrix = {}
    for a in airport_list:
        dist_matrix[a] = {}
        for b in airport_list:
            if a == b:
                dist_matrix[a][b] = 0
            else:
                try:
                    dist_matrix[a][b] = nx.shortest_path_length(G, a, b, weight='distance')
                except nx.NetworkXNoPath:
                    dist_matrix[a][b] = float('inf')
    return dist_matrix

def get_full_route(high_level_path, G):
    """Reconstruct actual flight path using shortest paths between high-level stops.
    Returns full sequence of airports (including any layovers within selected set)."""
    full_path = [high_level_path[0]]
    for i in range(1, len(high_level_path)):
        try:
            sub_path = nx.shortest_path(G, full_path[-1], high_level_path[i], weight='distance')
            full_path.extend(sub_path[1:])
        except nx.NetworkXNoPath:
            full_path.append(high_level_path[i])  # fallback
    return full_path