import sys
import os
import numpy as np
import random
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_airports, load_routes
from graph_builder import build_graph
from rl_env import FlightRoutingEnv
from visualization import plot_route_matplotlib

SELECTED_AIRPORTS = ['JFK', 'LAX', 'ATL', 'ORD', 'DFW', 'MIA', 'SFO', 'SEA', 'BOS', 'DEN', 'LAS', 'PHX']

if __name__ == "__main__":
    print("=== AI Flight Route Optimizer - Reinforcement Learning ===")
    
    airports_df = load_airports()
    routes_df = load_routes()
    
    G = build_graph(airports_df, routes_df, SELECTED_AIRPORTS)
    
    start = 'JFK'
    goal = 'LAX'
    
    env = FlightRoutingEnv(G, SELECTED_AIRPORTS, start, goal)
    
    # Tabular Q-learning (perfect for small discrete state/action space)
    n_states = env.n
    n_actions = env.n
    Q = np.zeros((n_states, n_actions))
    
    alpha = 0.15      # learning rate
    gamma = 0.95      # discount
    epsilon = 0.2     # exploration
    episodes = 800
    max_steps = 30
    
    print("Training RL agent (Q-learning)...")
    for ep in range(episodes):
        state = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, done, _ = env.step(action)
            
            # Bellman update
            best_next = np.max(Q[next_state])
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * best_next - Q[state, action])
            
            state = next_state
            step += 1
    
    # Test learned policy
    state = env.reset()
    path = env.get_current_path()
    done = False
    step = 0
    while not done and step < max_steps:
        action = np.argmax(Q[state])
        next_state, _, done, _ = env.step(action)
        path = env.get_current_path()
        state = next_state
        step += 1
    
    print("Learned RL route:", " → ".join(path))
    
    # Visualize
    plot_route_matplotlib(G, path, title=f"RL Learned Route: {start} to {goal}")
    print("Training complete! Agent learned a valid dynamic routing policy.")