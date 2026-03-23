import matplotlib as plt
import numpy as np
import pprint

class Simulator():
    n_robots                = None
    graph_edges             = None
    is_undirected           = None

    adjascency_list         = None
    graph_laplacian         = None
    algebraic_connectivity  = None

    def __init__(self, graph_edges: np.ndarray, n_robots: int, is_undirected: bool = False):
        self.n_robots           = n_robots
        self.graph_edges        = graph_edges
        self.is_undirected      = is_undirected

        self.adjascency_list    = self.get_adjacency_list()
        self.graph_laplacian    = self.get_graph_laplacian()
        self.algebraic_connectivity = self.get_algebraic_connectivity()

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
    
    def simulate_rendezvous(self):
        pass    

def main() -> None:
    edges = np.array(
        [
            [2, 3],
            [3, 1],
            [1, 2],
            [1, 3]
        ]
    )

    sim = Simulator(edges, 4)
    print(sim.n_robots)
    print(sim.graph_laplacian)
    print(sim.adjascency_list)
    print(sim.algebraic_connectivity)


if __name__ == '__main__':
    main()