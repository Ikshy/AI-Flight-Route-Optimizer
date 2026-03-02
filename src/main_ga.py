import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_airports, load_routes
from graph_builder import build_graph, get_distance_matrix, get_full_route
from ga_optimizer import genetic_algorithm
from visualization import plot_route_matplotlib

# Major US hubs - ensures connected subgraph and realistic routes
SELECTED_AIRPORTS = ['JFK', 'LAX', 'ATL', 'ORD', 'DFW', 'MIA', 'SFO', 'SEA', 'BOS', 'DEN', 'LAS', 'PHX']

if __name__ == "__main__":
    print("=== AI Flight Route Optimizer - Genetic Algorithm ===")
    
    airports_df = load_airports()
    routes_df = load_routes()
    
    G = build_graph(airports_df, routes_df, SELECTED_AIRPORTS)
    
    start = 'JFK'
    goal = 'LAX'
    intermediates = [a for a in SELECTED_AIRPORTS if a not in (start, goal)][:6]  # 6 stops for fast GA convergence
    
    dist_matrix = get_distance_matrix(G, SELECTED_AIRPORTS)
    
    print(f"Optimizing route: {start} → (visit {intermediates}) → {goal}")
    high_level_path, distance = genetic_algorithm(
        start, goal, intermediates, dist_matrix,
        pop_size=60, generations=80
    )
    
    full_path = get_full_route(high_level_path, G)
    print("Full optimized route:", " → ".join(full_path))
    print(f"Total distance: {distance:.1f} km")
    
    plot_route_matplotlib(G, full_path, title=f"GA Optimized Route: {start} to {goal} ({distance:.0f} km)")