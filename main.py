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
                
                if 'preset' in data:
                    preset = data['preset']
                    params = data.get('preset_params', {})
                    self.initial_robot_positions = self.generate_preset_positions(preset, params)
                    self.n_robots = len(self.initial_robot_positions)
                else:
                    self.initial_robot_positions = data['initial_robot_positions']
                    self.n_robots = len(self.initial_robot_positions)

                self.bias = data.get('bias', [])
                pos_array = np.array(self.initial_robot_positions, dtype=float)
                if not self.bias:
                    self.bias = np.zeros_like(pos_array)
                else:
                    self.bias = np.array(self.bias, dtype=float)
                    if self.bias.shape != pos_array.shape:
                        raise ValueError(f"Bias shape {self.bias.shape} does not match positions shape {pos_array.shape}")

                if str(data['rendezvous_radius']).upper() == 'INFINITY':
                    self.rendezvous_radius = math.inf
                else:
                    self.rendezvous_radius = float(data['rendezvous_radius'])
                
                fixed_raw = data.get('fixed_edges', [])
                prohibited_raw = data.get('prohibited_edges', [])
                
                max_index = self.n_robots - 1
                self.fixed_edges = []
                for u, v in fixed_raw:
                    if u <= max_index and v <= max_index:
                        self.fixed_edges.append((u, v) if u < v else (v, u))
                if len(self.fixed_edges) < len(fixed_raw):
                    print("Warning: some fixed edges were ignored due to invalid indices.")
                
                self.prohibited_edges = []
                for u, v in prohibited_raw:
                    if u <= max_index and v <= max_index:
                        self.prohibited_edges.append((u, v) if u < v else (v, u))
                if len(self.prohibited_edges) < len(prohibited_raw):
                    print("Warning: some prohibited edges were ignored due to invalid indices.")
                
                self.use_proximity = data.get('use_proximity', True)
                
                if not self.use_proximity and len(self.fixed_edges) == 0:
                    print("Warning: No edges defined and 'use_proximity' is off. Robots will not communicate!")
                
        except Exception as e:
            print(e)
            exit(1)
        
    def generate_preset_positions(self, preset, params):
        n = params.get('n_robots', 5)
        dim = params.get('dim', 2)

        center = list(params.get('center', [0]*dim))
        if len(center) < dim:
            center.extend([0] * (dim - len(center)))
        center = np.array(center[:dim])

        if n <= 0:
            raise ValueError("n_robots deve ser positivo")

        if preset == 'ring':
            radius = params.get('radius', 5.0)
            positions = []
            for i in range(n):
                angle = 2 * np.pi * i / n
                pos = [center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle)]
                if dim == 3:
                    pos.append(center[2])
                positions.append(pos)
            return positions

        elif preset == 'line':
            length = params.get('length', 10.0)
            start = center - length/2
            end   = center + length/2
            positions = [start + (end - start) * i / max(n-1, 1) for i in range(n)]
            return [p.tolist() for p in positions]

        elif preset == 'grid':
            rows = params.get('rows', int(np.sqrt(n)))
            cols = params.get('cols', int(np.ceil(n / rows)))
            spacing = params.get('spacing', 1.0)
            positions = []
            for i in range(rows):
                for j in range(cols):
                    if len(positions) >= n:
                        break
                    x = center[0] + (j - (cols-1)/2) * spacing
                    y = center[1] + (i - (rows-1)/2) * spacing
                    if dim == 3:
                        positions.append([x, y, center[2]])
                    else:
                        positions.append([x, y])
            return positions[:n]

        elif preset == 'sphere':
            radius = params.get('radius', 5.0)
            positions = []
            if n == 1:
                pos = [center[0], center[1] + radius] if dim == 2 else [center[0], center[1] + radius, center[2]]
                return [pos]
            phi = np.pi * (3. - np.sqrt(5.))
            for i in range(n):
                y = 1 - (i / (n - 1)) * 2
                radius_y = np.sqrt(1 - y*y)
                theta = phi * i
                x = np.cos(theta) * radius_y
                z = np.sin(theta) * radius_y
                pos = [center[0] + radius * x,
                    center[1] + radius * y,
                    center[2] + radius * z]
                if dim == 2:
                    pos = pos[:2]
                positions.append(pos)
            return positions

        elif preset == 'random':
            box_size = params.get('box_size', 10.0)
            low = center - box_size/2
            high = center + box_size/2
            positions = np.random.uniform(low, high, size=(n, dim)).tolist()
            return positions

        else:
            raise ValueError(f"Preset desconhecido: {preset}")
    
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
        edges_history = []
        
        for t in range(steps):
            for i in range(self.n_robots):
                history[i].append(curr_pos[i].copy())

            edges_set = set()

            if self.use_proximity:
                for i in range(self.n_robots):
                    for j in range(i + 1, self.n_robots):
                        dist = np.linalg.norm(curr_pos[i] - curr_pos[j])
                        if dist <= self.rendezvous_radius:
                            edges_set.add((i, j))

            for u, v in self.fixed_edges:
                if u < v:
                    edges_set.add((u, v))
                else:
                    edges_set.add((v, u))

            prohibited = set()
            if self.prohibited_edges:
                for u, v in self.prohibited_edges:
                    if u < v:
                        prohibited.add((u, v))
                    else:
                        prohibited.add((v, u))
                edges_set -= prohibited

            edges = [list(pair) for pair in edges_set]
            edges_history.append(edges)
            
            L = self.compute_laplacian(edges) 
            lambda2_history.append(self.get_algebraic_connectivity())

            velocity = -L.dot(curr_pos) + self.bias
            

            curr_pos += velocity * dt

        return np.array(history), np.array(lambda2_history), edges_history
    
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
    history, lambda2_history, edges_history = sim.simulate_rendezvous(dt=dt, steps=500)

    viz = Visualizer(history, dt, lambda2_history, edges_history)
    viz.plot_analysis()
    viz.animate()
    

if __name__ == '__main__':
    main()