import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import json

from matplotlib.animation import FuncAnimation

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

        if self.algebraic_connectivity is not None:
            return self.algebraic_connectivity

        eigenvalues = np.sort(np.linalg.eigvals(self.graph_laplacian))

        if len(eigenvalues) < 2:
            return 0
        
        return eigenvalues[1]
    
    def simulate_rendezvous(self, dt=0.01, steps=500):
        curr_pos = np.array(self.initial_robot_positions, dtype=float)
        history = [[] for _ in range(self.n_robots)]

        for t in range(steps):
            for i in range(self.n_robots):
                history[i].append(curr_pos[i].copy())

            edges = []
            for i in range(self.n_robots):
                for j in range(i + 1, self.n_robots):
                    dist = np.linalg.norm(curr_pos[i] -curr_pos[j])
                    if dist <= self.rendezvous_radius:
                        edges.append([i, j])
            
            self.graph_edges = edges
            L = self.compute_laplacian(edges) 
            velocity = -L.dot(curr_pos)
            curr_pos += velocity * dt

        return np.array(history)

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
        return L
    
    def plot_analysis(self, history, dt):
        """ Gera o gráfico de distância em relação ao ponto de encontro """
        steps = history.shape[1]
        time_axis = np.linspace(0, steps * dt, steps)
        
        start_pos = history[:, 0, :]
        meeting_point = np.mean(start_pos, axis=0)

        plt.figure(figsize=(10, 5))
        for i in range(self.n_robots):
            distances = [np.linalg.norm(history[i, t] - meeting_point) for t in range(steps)]
            plt.plot(time_axis, distances, label=f'Robô {i}')

        plt.title('Distância de cada robô até o Ponto de Encontro')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Distância')
        plt.grid(True)
        plt.legend()
        plt.show()

    def animate_rendezvous(self, history):
        """ Cria a animação dos robôs convergindo """
        fig, ax = plt.subplots(figsize=(7, 7))
        
        all_x = history[:, :, 0]
        all_y = history[:, :, 1]
        ax.set_xlim(np.min(all_x) - 1, np.max(all_x) + 1)
        ax.set_ylim(np.min(all_y) - 1, np.max(all_y) + 1)
        ax.set_title("Simulação de Rendezvous")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)

        scatters = ax.scatter([], [], c='blue', s=50)
        lines = [ax.plot([], [], '--', alpha=0.3)[0] for _ in range(self.n_robots)]

        def update(frame):
            for i in range(self.n_robots):
                lines[i].set_data(history[i, :frame, 0], history[i, :frame, 1])
            
            curr_step_pos = history[:, frame, :]
            scatters.set_offsets(curr_step_pos)
            return lines + [scatters]

        ani = FuncAnimation(fig, update, frames=history.shape[1], interval=20, blit=True)
        plt.show()


def main() -> None:

    sim = Simulator()
    dt = 0.01
    history = sim.simulate_rendezvous(dt=dt, steps=1000)
    sim.plot_analysis(history, dt)
    sim.animate_rendezvous(history)    

if __name__ == '__main__':
    main()