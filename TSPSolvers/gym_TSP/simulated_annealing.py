#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Written by Aman Anand on 2025-04-05
# This code applies Simulated Annealing to the Traveling Salesman Problem (TSP).
# -----------------------------------------------------------------------------

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio


class TSPSimAnnealer:
    def __init__(self, dist_map, start_temp=1000, temp_decay=0.995, end_temp=1):
        self.dist_map = dist_map
        self.city_count = len(dist_map)
        self.start_temp = start_temp
        self.temp_decay = temp_decay
        self.end_temp = end_temp
        self.city_locations = np.random.rand(self.city_count, 2)  # Fixed city coordinates

    def compute_total_distance(self, route):
        # Compute the full tour distance (including return to the starting city)
        return sum(self.dist_map[route[i], route[i + 1]] for i in range(len(route) - 1)) + \
               self.dist_map[route[-1], route[0]]

    def exchange_two_cities(self, route):
        new_route = route.copy()
        idx1, idx2 = np.random.choice(len(route), 2, replace=False)
        new_route[idx1], new_route[idx2] = new_route[idx2], new_route[idx1]
        return new_route

    def run_simulated_annealing(self):
        current_route = np.random.permutation(self.city_count)
        current_length = self.compute_total_distance(current_route)

        best_route, best_length = current_route.copy(), current_length
        temp = self.start_temp
        snapshots = []
        history = [current_length]

        start_time = time.time()
        MAX_DURATION = 600  # Timeout after 10 minutes
        print(f"🔄 Initial Route Length: {current_length:.2f}")
        print("🚀 Beginning Simulated Annealing Process...")

        iteration = 0
        best_iter = 0

        while temp > self.end_temp:
            if time.time() - start_time > MAX_DURATION:
                print("⏳ Maximum runtime reached. Exiting loop.")
                break

            candidate_route = self.exchange_two_cities(current_route)
            candidate_length = self.compute_total_distance(candidate_route)
            delta_length = candidate_length - current_length

            # Accept the candidate route if it's better or by a probabilistic chance
            if delta_length < 0 or np.random.rand() < np.exp(-delta_length / temp):
                current_route, current_length = candidate_route, candidate_length
                history.append(current_length)

            if current_length < best_length:
                best_route, best_length = current_route.copy(), current_length
                best_iter = iteration

            if iteration % 10 == 0:
                frame = self.render_route(current_route, current_length, temp, iteration)
                snapshots.append(frame)

            temp *= self.temp_decay
            iteration += 1

        total_runtime = time.time() - start_time
        self.create_gif(snapshots, "simulated_annealing_tsp.gif")
        final_reward = -best_length

        print(f"🏁 Process Finished | Best Route Length: {best_length:.2f}, Total Time: {total_runtime:.2f}s, Iterations: {iteration}")
        return best_route, best_length, total_runtime, best_iter, final_reward, history

    def render_route(self, route, route_length, temp, iter_num):
        fig, ax = plt.subplots(figsize=(5, 5))
        route_coords = self.city_locations[route]

        ax.plot(route_coords[:, 0], route_coords[:, 1], 'o-', color='darkorange', markersize=6)
        ax.plot([route_coords[-1, 0], route_coords[0, 0]],
                [route_coords[-1, 1], route_coords[0, 1]], 'r--', lw=1.5)
        ax.set_title(f"Iteration {iter_num}\nLength: {route_length:.2f}, Temp: {temp:.2f}", fontsize=10)
        ax.axis('off')
        fig.tight_layout()

        temp_file = "temp_sa_frame.png"
        plt.savefig(temp_file)
        plt.close(fig)
        return imageio.imread(temp_file)

    def create_gif(self, frames, filename):
        if not frames:
            print("⚠️ No frames available for GIF creation.")
            return
        try:
            imageio.mimsave(filename, frames, duration=0.2)
            print(f"✅ GIF successfully created: {filename}")
        except Exception as error:
            print(f"❌ Failed to create GIF: {error}")

# --- Example Usage ---
if __name__ == "__main__":
    n_cities = 126
    # Generate a random distance matrix with values between 10 and 100, and zeros on the diagonal.
    dist_matrix = np.random.randint(10, 100, size=(n_cities, n_cities))
    np.fill_diagonal(dist_matrix, 0)

    annealer = TSPSimAnnealer(dist_matrix)
    route, length, run_time, best_iter, reward, route_history = annealer.run_simulated_annealing()

    print(f"🔹 Optimal Route Length: {length:.2f}")
    print(f"⏳ Total Runtime: {run_time:.2f} seconds")
    print(f"📉 Best Improvement at Iteration: {best_iter}")
    print(f"🏆 Final Reward: {reward}")
    print(f"📈 Route History: {route_history}")

