#!/usr/bin/env python
# -----------------------------------------------------------------------------
# written by Aman Anand on 2025-04-05
# This script applies the Iterative Deepening A* (IDA*) algorithm to solve the
# Frozen Lake environment and creates a GIF to visualize the solution path.
# -----------------------------------------------------------------------------

import gymnasium as gym
import numpy as np
import imageio
import cv2  # OpenCV for drawing arrows
import matplotlib.pyplot as plt
import time  # For timing execution

class IDAStarPathfinder:
    def __init__(self, environment):
        self.env = environment
        self.initial_state = 0
        self.target_state = environment.observation_space.n - 1
        self.optimal_path = None
        self.optimal_cost = float("inf")
        self.runtime = None  # To store execution time

    def manhattan_heuristic(self, state):
        """Compute Manhattan distance as the heuristic."""
        grid_dim = int(np.sqrt(self.env.observation_space.n))
        x1, y1 = divmod(state, grid_dim)
        x2, y2 = divmod(self.target_state, grid_dim)
        return abs(x1 - x2) + abs(y1 - y2)

    def recursive_search(self, state, path, cost, threshold):
        """Recursive depth-first search with threshold pruning."""
        f = cost + self.manhattan_heuristic(state)
        if f > threshold:
            return f  # New minimum threshold candidate

        if state == self.target_state:
            self.optimal_path = path
            self.optimal_cost = cost
            return None  # Solution found

        min_threshold = float("inf")
        for action in range(self.env.action_space.n):
            for prob, next_state, _, _ in self.env.unwrapped.P[state][action]:
                if prob > 0 and next_state not in path:
                    result = self.recursive_search(next_state, path + [next_state], cost + 1, threshold)
                    if result is None:
                        return None  # Propagate solution found
                    min_threshold = min(min_threshold, result)
        return min_threshold

    def iterative_deepening_a_star(self):
        """Perform IDA* search."""
        start_timer = time.perf_counter()
        current_threshold = self.manhattan_heuristic(self.initial_state)

        while True:
            result = self.recursive_search(self.initial_state, [self.initial_state], 0, current_threshold)
            if result is None:
                self.runtime = time.perf_counter() - start_timer
                return self.optimal_path, self.optimal_cost, self.runtime
            if result == float("inf"):
                self.runtime = time.perf_counter() - start_timer
                return None, float("inf"), self.runtime
            current_threshold = result

    def create_animation(self, path, gif_filename="ida_frozen_lake.gif"):
        """Generate a GIF showing the agent's movement along the solution path."""
        frames = []
        obs, _ = self.env.reset()

        for i in range(len(path) - 1):
            current = path[i]
            nxt = path[i + 1]

            # Determine the correct action to move from current to next state.
            for action in range(self.env.action_space.n):
                for prob, state, reward, done in self.env.unwrapped.P[current][action]:
                    if state == nxt and prob > 0:
                        obs, _, _, _, _ = self.env.step(action)
                        break

            frame = self.env.render()
            frames.append(frame)

        imageio.mimsave(gif_filename, frames, duration=0.5)
        print(f"✅ GIF saved as {gif_filename}")

        # Optionally display the final frame
        plt.imshow(frames[-1])
        plt.axis("off")
        # plt.show()  # Uncomment to display the image window

    def render_frozen_lake_state(self, agent_state, previous_state=None):
        """Render an image of the Frozen Lake with directional arrows."""
        grid_dim = int(np.sqrt(self.env.observation_space.n))
        frozen_map = self.env.unwrapped.desc
        canvas = np.ones((grid_dim * 100, grid_dim * 100, 3), dtype=np.uint8) * 255

        color_codes = {
            b"S": (0, 255, 0),    # Start
            b"F": (200, 200, 200),# Frozen
            b"H": (0, 0, 0),      # Hole
            b"G": (255, 215, 0)   # Goal
        }

        # Draw the grid map.
        for i in range(grid_dim):
            for j in range(grid_dim):
                tile = frozen_map[i][j]
                canvas[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100] = color_codes[tile]

        # Draw the agent.
        agent_x, agent_y = divmod(agent_state, grid_dim)
        cv2.circle(canvas, (agent_y * 100 + 50, agent_x * 100 + 50), 30, (255, 0, 0), -1)

        # Draw an arrow from the previous position if available.
        if previous_state is not None:
            prev_x, prev_y = divmod(previous_state, grid_dim)
            start_pt = (prev_y * 100 + 50, prev_x * 100 + 50)
            end_pt = (agent_y * 100 + 50, agent_x * 100 + 50)
            canvas = cv2.arrowedLine(canvas, start_pt, end_pt, (0, 0, 255), 5)

        return canvas

# Initialize the Frozen Lake environment (non-slippery version)
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

# Solve the environment using the IDA* algorithm
pathfinder = IDAStarPathfinder(env)
solution_path, solution_cost, exec_time = pathfinder.iterative_deepening_a_star()

if solution_path:
    print("Best Path:", solution_path)
    print("Best Cost:", solution_cost)
    print(f"Execution Time: {exec_time:.6f} seconds")
    pathfinder.create_animation(solution_path)
else:
    print("No solution found.")

