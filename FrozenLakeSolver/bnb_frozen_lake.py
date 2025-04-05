#!/usr/bin/env python
# -----------------------------------------------------------------------------
# written by aman anand on 2025-04-05
# This script uses a Branch and Bound approach to solve the Frozen Lake environment
# and creates an animation (GIF) of the solution path.
# -----------------------------------------------------------------------------

import gymnasium as gym
import numpy as np
import heapq
import imageio
import cv2  # OpenCV for drawing arrows
import matplotlib.pyplot as plt
import time  # For measuring execution time

class BnBPathSolver:
    def __init__(self, env):
        self.env = env
        self.initial_state = 0
        self.final_state = env.observation_space.n - 1
        self.optimal_cost = float("inf")
        self.optimal_path = None
        self.run_time = None  # To store the execution time

    def run_branch_and_bound(self):
        """Solve the Frozen Lake environment using a Branch and Bound approach."""
        start = time.perf_counter()
        # Priority queue: elements are (current_cost, current_path)
        queue = [(0, [self.initial_state])]
        heapq.heapify(queue)

        while queue:
            current_cost, current_path = heapq.heappop(queue)
            current_node = current_path[-1]

            if current_node == self.final_state:
                self.optimal_cost = current_cost
                self.optimal_path = current_path
                self.run_time = time.perf_counter() - start
                return self.optimal_path, self.optimal_cost, self.run_time

            # Expand node by trying all possible actions
            for action in range(self.env.action_space.n):
                for prob, next_node, reward, done in self.env.unwrapped.P[current_node][action]:
                    # Proceed if transition is valid and avoids cycles
                    if prob > 0 and next_node not in current_path:
                        new_cost = current_cost + 1
                        if new_cost < self.optimal_cost:
                            heapq.heappush(queue, (new_cost, current_path + [next_node]))

        self.run_time = time.perf_counter() - start
        return None, float("inf"), self.run_time

    def create_animation(self, path, filename="bnb_frozen_lake.gif"):
        """Generates a GIF showing the agent's journey on the Frozen Lake."""
        frames = []
        obs, _ = self.env.reset()

        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]

            # Identify the action that moves from current_node to next_node
            for action in range(self.env.action_space.n):
                for prob, state, reward, done in self.env.unwrapped.P[current_node][action]:
                    if state == next_node and prob > 0:
                        obs, _, _, _, _ = self.env.step(action)
                        break

            frame = self.env.render()
            frames.append(frame)

        imageio.mimsave(filename, frames, duration=0.5)
        print(f"✅ GIF saved as {filename}")

        # Optionally display the final frame
        plt.imshow(frames[-1])
        plt.axis("off")
        # plt.show()

    def render_environment(self, agent_state, prev_state=None):
        """Creates an image of the Frozen Lake with movement arrows."""
        grid_size = int(np.sqrt(self.env.observation_space.n))
        lake_layout = self.env.unwrapped.desc
        image = np.ones((grid_size * 100, grid_size * 100, 3), dtype=np.uint8) * 255  # White background

        color_dict = {
            b"S": (0, 255, 0),      # Start: Green
            b"F": (200, 200, 200),  # Frozen: Light gray
            b"H": (0, 0, 0),        # Hole: Black
            b"G": (255, 215, 0)     # Goal: Yellow
        }

        # Draw the map grid
        for i in range(grid_size):
            for j in range(grid_size):
                tile = lake_layout[i][j]
                image[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100] = color_dict[tile]

        # Draw the agent on the grid
        agent_x, agent_y = divmod(agent_state, grid_size)
        cv2.circle(image, (agent_y * 100 + 50, agent_x * 100 + 50), 30, (255, 0, 0), -1)  # Red circle

        # Draw an arrow if there is a previous state
        if prev_state is not None:
            prev_x, prev_y = divmod(prev_state, grid_size)
            start_pt = (prev_y * 100 + 50, prev_x * 100 + 50)
            end_pt = (agent_y * 100 + 50, agent_x * 100 + 50)
            image = cv2.arrowedLine(image, start_pt, end_pt, (0, 0, 255), 5)  # Red arrow

        return image

# Initialize Frozen Lake environment (non-slippery)
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

# Solve using Branch and Bound approach
bnb_solver = BnBPathSolver(env)
solution_path, solution_cost, exec_time = bnb_solver.run_branch_and_bound()

if solution_path:
    print("Best Path:", solution_path)
    print("Best Cost:", solution_cost)
    print(f"Execution Time: {exec_time:.6f} seconds")
    bnb_solver.create_animation(solution_path)
else:
    print("No solution found.")

