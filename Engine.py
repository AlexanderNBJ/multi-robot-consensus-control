import numpy as np
import math

class ConsensusController:
    def __init__(self, offsets=None, bias=None, angular_velocity=0.0, rotation_center_index=None, relative_offsets=None):
        self.offsets = np.array(offsets) if offsets is not None else None
        self.bias = np.array(bias) if bias is not None else None
        self.omega = angular_velocity
        self.rotation_center_index = rotation_center_index
        self.relative_offsets = np.array(relative_offsets) if relative_offsets is not None else None
        self.angle = 0.0

    def compute_velocities(self, positions, laplacian, dt):
        n, dim = positions.shape
        if self.bias is None:
            self.bias = np.zeros_like(positions)

        if self.omega != 0.0:
            if self.rotation_center_index is not None:
                center = positions[self.rotation_center_index]
            else:
                center = np.mean(positions, axis=0)
        else:
            center = np.zeros(dim)

        if self.omega != 0.0 and self.relative_offsets is not None:
            self.angle += self.omega * dt
            self.angle = self.angle % (2 * np.pi)
            cos_a = np.cos(self.angle)
            sin_a = np.sin(self.angle)
            targets = center + self._rotate_offsets(self.relative_offsets, cos_a, sin_a)
        elif self.offsets is not None:
            targets = self.offsets
        else:
            targets = np.tile(center, (n, 1))

        velocity = -laplacian.dot(positions - targets) + self.bias

        return velocity

    def _rotate_offsets(self, rel_offsets, cos_a, sin_a):
        rotated = rel_offsets.copy()
        
        if rel_offsets.shape[1] >= 2:
            x = rel_offsets[:, 0]
            y = rel_offsets[:, 1]
            rotated[:, 0] = x * cos_a - y * sin_a
            rotated[:, 1] = x * sin_a + y * cos_a
        return rotated
    
class NetworkTopology:
    def __init__(self, use_proximity=True, rendezvous_radius=1.0, fixed_edges=None, prohibited_edges=None):
        self.use_proximity = use_proximity
        self.rendezvous_radius = rendezvous_radius
        self.fixed_edges = fixed_edges or []
        self.prohibited_edges = prohibited_edges or []

    def compute_edges_and_laplacian(self, positions):
        n_robots = len(positions)
        edges_set = set()

        for u, v in self.fixed_edges:
            edges_set.add((min(u, v), max(u, v)))

        if self.use_proximity:
            for i in range(n_robots):
                for j in range(i + 1, n_robots):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist <= self.rendezvous_radius:
                        edges_set.add((i, j))

        for u, v in self.prohibited_edges:
            edge = (min(u, v), max(u, v))
            edges_set.discard(edge)

        edges = list(edges_set)
        
        L = np.zeros([n_robots, n_robots])
        for u, v in edges:
            L[u][v] = -1
            L[v][u] = -1
            L[u][u] += 1
            L[v][v] += 1

        return edges, L
    
    def get_algebraic_connectivity(self, laplacian):
        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues.sort()
        return eigenvalues[1] if len(eigenvalues) >= 2 else 0.0

class SimulatorEngine:
    def __init__(self, initial_positions, topology: NetworkTopology, controller: ConsensusController):
        self.positions = np.array(initial_positions, dtype=float)
        self.n_robots, self.dim = self.positions.shape
        
        self.topology = topology
        self.controller = controller

    def run(self, dt=0.01, steps=500):
        history = np.zeros((self.n_robots, steps, self.dim))
        lambda2_history = np.zeros(steps)
        edges_history = []

        curr_pos = self.positions.copy()

        for t in range(steps):
            history[:, t, :] = curr_pos
            
            edges, L = self.topology.compute_edges_and_laplacian(curr_pos)
            edges_history.append(edges)
            lambda2_history[t] = self.topology.get_algebraic_connectivity(L)
            velocities = self.controller.compute_velocities(curr_pos, L, dt)

            curr_pos += velocities * dt

        return history, lambda2_history, edges_history

def build_simulator_from_dict(data: dict) -> SimulatorEngine:
    initial_positions = data.get('initial_robot_positions', [])
    
    topology = NetworkTopology(
        use_proximity=data.get('use_proximity', True),
        rendezvous_radius=data.get('rendezvous_radius', math.inf),
        fixed_edges=data.get('fixed_edges', []),
        prohibited_edges=data.get('prohibited_edges', [])
    )
    
    controller = ConsensusController(
        offsets=np.array(data.get('formation_offsets')) if 'formation_offsets' in data else None,
        bias=np.array(data.get('bias')) if 'bias' in data else None
    )
    
    return SimulatorEngine(initial_positions, topology, controller)