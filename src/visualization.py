import matplotlib.pyplot as plt
import networkx as nx

def plot_route_matplotlib(G, full_path, title="AI Optimized Flight Route"):
    """Matplotlib scatter plot of airports + route lines.
    Simple but clear geographic visualization (lon/lat projection)."""
    plt.figure(figsize=(14, 8))
    
    # All airports in graph
    nodes = list(G.nodes())
    lons = [G.nodes[n]['lon'] for n in nodes]
    lats = [G.nodes[n]['lat'] for n in nodes]
    plt.scatter(lons, lats, color='skyblue', s=80, edgecolors='black', zorder=2)
    
    # Labels
    for n in nodes:
        plt.annotate(n, (G.nodes[n]['lon'] + 0.5, G.nodes[n]['lat'] + 0.3),
                     fontsize=9, fontweight='bold')
    
    # Route
    path_lons = [G.nodes[p]['lon'] for p in full_path]
    path_lats = [G.nodes[p]['lat'] for p in full_path]
    plt.plot(path_lons, path_lats, 'r-', linewidth=3, alpha=0.8, zorder=1, label='Route')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()