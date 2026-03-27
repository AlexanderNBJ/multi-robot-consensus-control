import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import math
import json
from Visualizer import Visualizer

class Simulator():
    n_robots                = None
    graph_edges             = None
    is_undirected           = None

    adjascency_list         = None
    graph_laplacian         = None
    algebraic_connectivity  = None

    def __init__(self):
        try:
            with open('config.json') as file:
                data = json.load(file)

                self.n_robots = len(data['initial_robot_positions'])
                self.initial_robot_positions = data['initial_robot_positions']

                if data['rendezvous_radius'] == 'INFINITY':
                    self.rendezvous_radius = math.inf
                else:
                    self.rendezvous_radius = data['rendezvous_radius']
                
        except Exception as e:
            print(e)
            exit(1)

    def get_graph_laplacian(self):

        if self.graph_laplacian is not None:
            return self.graph_laplacian

        robots_quantity = len(self.adjascency_list)
        laplacian = np.zeros([robots_quantity, robots_quantity])

        for i in range(robots_quantity):
            laplacian[i][i] = len(self.adjascency_list[i])

            for neighbor in self.adjascency_list[i]:
                laplacian[i][neighbor] = -1
        return laplacian

    def get_adjacency_list(self):

        if self.adjascency_list is not None:
            return self.adjascency_list

        adjacency_list = {i: [] for i in range(0, self.n_robots)}

        for u, v in self.graph_edges:
            if v not in adjacency_list[u]:
                adjacency_list[u].append(v)

            if u not in adjacency_list[v] and self.is_undirected:
                adjacency_list[v].append(u)

        return adjacency_list

    def get_algebraic_connectivity(self):
        if self.graph_laplacian is None:
            return 0
        
        eigenvalues = np.linalg.eigvalsh(self.graph_laplacian)
        eigenvalues.sort()

        if len(eigenvalues) < 2:
            return 0
        
        return eigenvalues[1]
    
    def simulate_rendezvous(self, dt=0.01, steps=500):
        curr_pos = np.array(self.initial_robot_positions, dtype=float)
        history = [[] for _ in range(self.n_robots)]
        lambda2_history = []
        
        for t in range(steps):
            for i in range(self.n_robots):
                history[i].append(curr_pos[i].copy())

            # creates the proximity graph based on the rendezvous_radius
            edges = []
            for i in range(self.n_robots):
                for j in range(i + 1, self.n_robots):
                    dist = np.linalg.norm(curr_pos[i] -curr_pos[j])
                    if dist <= self.rendezvous_radius:
                        edges.append([i, j])
            
            # creates the laplacian matrix based on the proximity graph generated
            self.graph_edges = edges
            L = self.compute_laplacian(edges) 
            
            lambda2_history.append(self.get_algebraic_connectivity())
            # current velocity of each robot
            velocity = -L.dot(curr_pos)
            
            # current position based on a approximation of velocity integral
            curr_pos += velocity * dt

        return np.array(history), np.array(lambda2_history)

    def compute_laplacian(self, edges):
        n = self.n_robots
        adj = {i: [] for i in range(n)}

        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        L = np.zeros([n, n])

        for i in range(n):
            L[i][i] = len(adj[i])

            for neighbor in adj[i]:
                L[i][neighbor] = -1
        
        self.graph_laplacian = L
        return L

def main():
    sim = Simulator()
    dt = 0.01
    history, lambda2_history = sim.simulate_rendezvous(dt=dt, steps=100)

    viz = Visualizer(history, dt, lambda2_history)
    viz.plot_analysis()
    viz.animate()
    

if __name__ == '__main__':
    main()