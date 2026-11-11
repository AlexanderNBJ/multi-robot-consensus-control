import numpy as np

class ConsensusProtocol:
    """
    Protocolo de controle cooperativo distribuído.
    Implementa as leis de consenso P, PD ou PID descritas em Chen & Ren (2019) e Olfati-Saber (2007).
    """
    def __init__(self, kp=1.0, ki=0.0, kd=1.0, formation_offsets=None, input_bias=None, omega=0.0, rotation_center=None, relative_offsets=None, use_atan=False):
        # Ganhos locais do Controlador Cooperativo (K_local)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Vetores de controle de formação (desvios de posição p_ij)
        self.p_star = np.array(formation_offsets) if formation_offsets is not None else None
        self.u_0 = np.array(input_bias) if input_bias is not None else None # Viés externo do líder/grupo
        
        self.omega = omega
        self.rotation_center = rotation_center
        self.relative_offsets = np.array(relative_offsets) if relative_offsets is not None else None
        self.angle = 0.0
        
        # Parâmetro de saturação por arcotangente
        self.use_atan = use_atan

    def _compute_desired_states(self, X, n, dim, dt):
        """Calcula as posições desejadas p* (formation shape) para rastreamento."""
        if self.omega != 0.0:
            center = X[self.rotation_center] if self.rotation_center is not None else np.mean(X, axis=0)
        else:
            center = np.zeros(dim)

        if self.omega != 0.0 and self.relative_offsets is not None:
            self.angle = (self.angle + self.omega * dt) % (2 * np.pi)
            rotated_p = self.relative_offsets.copy()
            if self.relative_offsets.shape[1] >= 2:
                x, y = self.relative_offsets[:, 0], self.relative_offsets[:, 1]
                rotated_p[:, 0] = x * np.cos(self.angle) - y * np.sin(self.angle)
                rotated_p[:, 1] = x * np.sin(self.angle) + y * np.cos(self.angle)
            return center + rotated_p
        elif self.p_star is not None:
            return self.p_star
        else:
            return np.tile(center, (n, 1))

    def compute_control_input(self, X, V, Integral_Err, L, dt):
        """
        Calcula o vetor de entradas U = [u_1, ..., u_n]^T.
        Lei de controle (Olfati-Saber Eq. 7 / Chen & Ren Eq. 4.11 / Eq. 6.8).
        """
        B_ext = self.u_0 if self.u_0 is not None else np.zeros_like(X)

        X_target = self._compute_desired_states(X, len(X), X.shape[1], dt)
        
        # Erro de formação (Chen & Ren Eq. 4.24: \tilde{x}_i = x_i - x_i^*)
        X_tilde = X - X_target

        # Componente Proporcional: u_P = -k_p * L * \tilde{X} 
        U_P = self.kp * L.dot(X_tilde)
        
        # Componente Derivativa (Amortecimento Relativo): u_D = -k_d * L * V 
        U_D = self.kd * L.dot(V) if V is not None else 0
        
        # Componente Integral (Rastreamento de Erro Estacionário)
        U_I = self.ki * Integral_Err
        
        # Lei de Controle Cooperativa Global
        U = - U_P - U_D - U_I + B_ext
        
        # Saturação via arcotangente [-1, 1]
        if self.use_atan:
            U = (2.0 / np.pi) * np.arctan(U)
            
        return U
    
class InteractionGraph:
    """
    Modela a topologia da rede G = (V, E) e extrai a matriz Laplaciana L = D - A.
    Baseado nas formulações algébricas de Olfati-Saber et al. (2007).
    """
    def __init__(self, use_proximity=True, interaction_radius=1.0, fixed_edges=None, prohibited_edges=None):
        self.use_proximity = use_proximity
        self.r = interaction_radius # Raio de comunicação r > 0
        self.E_fixed = fixed_edges or []
        self.E_prohibited = prohibited_edges or []

    def compute_graph_matrices(self, X):
        """
        Gera a Matriz de Adjacência A e computa a Laplaciana L = D - A.
        (Olfati-Saber Eq. 9).
        """
        n = len(X)
        edges_set = set(tuple(sorted((u, v))) for u, v in self.E_fixed)

        # Matriz de Adjacência A_ij baseada em proximidade espacial (Chen & Ren Eq. 4.17)
        if self.use_proximity:
            for i in range(n):
                for j in range(i + 1, n):
                    if np.linalg.norm(X[i] - X[j]) <= self.r:
                        edges_set.add((i, j))

        for u, v in self.E_prohibited:
            edges_set.discard(tuple(sorted((u, v))))

        E = list(edges_set)
        
        # Construção rigorosa: L = D - A
        A = np.zeros([n, n])
        for u, v in E:
            A[u][v] = A[v][u] = 1 # Grafo não direcionado (Simétrico)
            
        D = np.diag(np.sum(A, axis=1)) # Matriz de Graus
        L = D - A # Grafo Laplaciano (Olfati-Saber Eq. 9)

        return E, L
    
    def get_algebraic_connectivity(self, L):
        """
        Calcula lambda_2 (Conectividade Algébrica). 
        Se lambda_2 > 0, o grafo é conexo (Olfati-Saber Teorema 3).
        """
        eigenvalues = np.sort(np.linalg.eigvalsh(L))
        return float(np.real(eigenvalues[1])) if len(eigenvalues) >= 2 else 0.0

class MultiAgentSimulator:
    """
    Engine de Simulação MIMO. Processa a dinâmica das plantas em tempo discreto.
    """
    def __init__(self, initial_X, topology: InteractionGraph, protocol: ConsensusProtocol, dynamics_order=1):
        self.X = np.array(initial_X, dtype=float) # Matriz de Posições
        self.n, self.dim = self.X.shape
        self.V = np.zeros_like(self.X) # Matriz de Velocidades
        self.Integral_X = np.zeros_like(self.X) # Acúmulo Integral do Consenso
        
        self.topology = topology
        self.protocol = protocol
        self.dynamics_order = dynamics_order

    def run(self, dt=0.01, steps=500):
        # Históricos de espaço de estado
        X_history = np.zeros((self.n, steps, self.dim))
        V_history = np.zeros((self.n, steps, self.dim))
        lambda2_history = np.zeros(steps)
        edges_history = []

        for t in range(steps):
            X_history[:, t, :] = self.X
            V_history[:, t, :] = self.V
            
            # 1. Obtém o grafo dinâmico G(t) e sua Laplaciana L(t)
            E, L = self.topology.compute_graph_matrices(self.X)
            edges_history.append(E)
            lambda2_history[t] = self.topology.get_algebraic_connectivity(L)

            # 2. Computa o Erro de Consenso Coletivo (L * \tilde{X})
            X_target = self.protocol._compute_desired_states(self.X, self.n, self.dim, dt)
            consensus_error = L.dot(self.X - X_target) 
            self.Integral_X += consensus_error * dt # Integração temporal do erro

            # 3. Calcula o Sinal de Controle U(t)
            V_feedback = self.V if self.dynamics_order == 2 else None
            U = self.protocol.compute_control_input(self.X, V_feedback, self.Integral_X, L, dt)

            # 4. Atualização das Dinâmicas das Plantas via Integração de Euler
            if self.dynamics_order == 2:
                # Duplo Integrador: \dot{X} = V, \dot{V} = U (Chen & Ren Eq. 4.10)
                self.V += U * dt
                self.X += self.V * dt
            else:
                # Simples Integrador: \dot{X} = U (Chen & Ren Eq. 3.2)
                self.X += U * dt

        return X_history, V_history, lambda2_history, edges_history

    def get_global_state_space_matrix(self, L):
        """
        Formulação Algébrica do Espaço de Estados (MIMO).
        Adapta-se automaticamente para Primeira Ordem (Cinemática) ou Segunda Ordem (Inércia).
        """
        n = self.n
        kp, kd = self.protocol.kp, self.protocol.kd
        
        if self.dynamics_order == 1:
            # Dinâmica Local de 1ª Ordem (Integrador Simples): u = -kp*x
            A_local = np.array([[0]])
            B_local = np.array([[1]])
            K_local = np.array([[kp]]) 
            
        elif self.dynamics_order == 2:
            # Dinâmica Local de 2ª Ordem (Integrador Duplo): u = -kp*x - kd*v
            A_local = np.array([
                [0, 1],
                [0, 0]
            ])
            B_local = np.array([
                [0], 
                [1]
            ])
            K_local = np.array([[kp, kd]])
            
        else:
            raise ValueError(f"Ordem {self.dynamics_order} não suportada para análise algébrica baseada nos artigos.")
        
        # Expansão para a Rede Global
        I_n = np.eye(n)
        A_open = np.kron(I_n, A_local)
        
        # Matriz de Acoplamento do Grafo
        B_K_global = np.kron(L, B_local @ K_local)
        
        # Matriz de Malha Fechada
        A_cl = A_open - B_K_global
        
        return A_cl
