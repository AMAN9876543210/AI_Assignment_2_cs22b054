#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Modified by Jane Smith on 2025-04-05
# This code implements a hill-climbing approach to the Traveling Salesman Problem (TSP).
# -----------------------------------------------------------------------------

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio


class TSPHillClimber:
    def __init__(self, dist_matrix, restarts=20, iterations=1000):
        self.dist_matrix = dist_matrix
        self.total_cities = len(dist_matrix)
        self.restarts = restarts
        self.iterations = iterations
        self.coordinates = np.random.rand(self.total_cities, 2)  # Fixed positions for cities

    def route_length(self, route):
        # Sum the distance for the tour, returning to the start
        return sum(self.dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)) + \
               self.dist_matrix[route[-1], route[0]]

    def swap_cities(self, route):
        new_route = route.copy()
        a, b = np.random.choice(len(route), 2, replace=False)
        new_route[a], new_route[b] = new_route[b], new_route[a]
        return new_route

    def run_hill_climbing(self):
        optimal_route, optimal_length, convergence_history = None, float('inf'), []
        frames_list = []
        start_time = time.time()
        TIMEOUT = 600  # 10 minutes timeout

        print("🚀 Starting Hill-Climbing TSP...")

        for trial in range(self.restarts):
            current_route = np.random.permutation(self.total_cities)
            current_length = self.route_length(current_route)
            history = [current_length]

            print(f"🔄 Trial {trial+1} | Initial Length: {current_length:.2f}")

            for step in range(self.iterations):
                if time.time() - start_time > TIMEOUT:
                    print("⏳ Timeout reached. Exiting early.")
                    break

                candidate_route = self.swap_cities(current_route)
                candidate_length = self.route_length(candidate_route)

                if candidate_length < current_length:
                    current_route, current_length = candidate_route, candidate_length
                    history.append(candidate_length)

                if step % 10 == 0:
                    frame = self.visualize_route(current_route, step, current_length)
                    frames_list.append(frame)

            if current_length < optimal_length:
                optimal_route, optimal_length, convergence_history = current_route, current_length, history

        total_runtime = time.time() - start_time
        self.export_gif(frames_list, "hill_climbing_tsp.gif")
        convergence_point = len(convergence_history)
        reward = -optimal_length

        return optimal_route, optimal_length, total_runtime, convergence_point, reward, convergence_history

    def visualize_route(self, route, step, length):
        fig, ax = plt.subplots(figsize=(5, 5))
        ordered_coords = self.coordinates[route]

        ax.plot(ordered_coords[:, 0], ordered_coords[:, 1], 'o-', color='steelblue', markersize=6)
        ax.plot([ordered_coords[-1, 0], ordered_coords[0, 0]],
                [ordered_coords[-1, 1], ordered_coords[0, 1]], 'r--', lw=1.5)
        ax.set_title(f"Step {step}\nLength: {length:.2f}", fontsize=10)
        ax.axis('off')
        fig.tight_layout()

        temp_filename = "temp_frame.png"
        plt.savefig(temp_filename)
        plt.close(fig)
        return imageio.imread(temp_filename)

    def export_gif(self, frames, filename):
        if not frames:
            print("⚠️ No frames available to export.")
            return
        try:
            imageio.mimsave(filename, frames, duration=0.2)
            print(f"✅ GIF exported: {filename}")
        except Exception as err:
            print(f"❌ Error exporting GIF: {err}")

# --- Example Usage ---
if __name__ == "__main__":
    n_cities = 126
    # Create a random symmetric distance matrix with zeros on the diagonal.
    dist_matrix = np.random.randint(10, 100, size=(n_cities, n_cities))
    np.fill_diagonal(dist_matrix, 0)

    tsp_solver = TSPHillClimber(dist_matrix)
    route, length, runtime, convergence, reward, history = tsp_solver.run_hill_climbing()

    print(f"🔹 Optimal Length: {length:.2f}")
    print(f"⏳ Runtime: {runtime:.2f} sec")
    print(f"📈 Convergence Steps: {convergence}")
    print(f"🏆 Reward: {reward}")
    print(f"💰 History: {history}")

