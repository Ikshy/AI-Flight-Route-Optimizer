import gym
from gym import spaces
import numpy as np

class FlightRoutingEnv(gym.Env):
    """Gym environment for dynamic flight routing.
    State = current airport index.
    Action = choose next airport (0..N-1).
    Reward = -distance (or penalty for invalid move).
    Goal = reach destination with positive reward."""
    
    def __init__(self, G, airport_list, start_iata, goal_iata):
        super().__init__()
        self.G = G
        self.airport_list = airport_list
        self.n = len(airport_list)
        self.start_idx = airport_list.index(start_iata)
        self.goal_idx = airport_list.index(goal_iata)
        
        self.observation_space = spaces.Discrete(self.n)
        self.action_space = spaces.Discrete(self.n)
        
        self.current_idx = self.start_idx
        self.path = [start_iata]

    def reset(self):
        self.current_idx = self.start_idx
        self.path = [self.airport_list[self.start_idx]]
        return self.current_idx

    def step(self, action):
        next_idx = int(action)
        next_iata = self.airport_list[next_idx]
        current_iata = self.airport_list[self.current_idx]
        
        reward = -5.0  # small step cost
        done = False
        info = {}

        if self.G.has_edge(current_iata, next_iata):
            dist = self.G[current_iata][next_iata]['distance']
            reward = -dist / 50.0  # scale for learning
            self.current_idx = next_idx
            self.path.append(next_iata)
            
            if next_idx == self.goal_idx:
                reward += 200.0  # big success reward
                done = True
        else:
            reward = -50.0  # heavy penalty for invalid flight
            # stay put

        return self.current_idx, reward, done, info

    def get_current_path(self):
        return self.path