import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

class Visualizer:
    def __init__(self, history, vel_history, dt, lambda2_history=None, edges_history=None, target_fps=10, damping=None, engine=None):
        self.history = history
        self.vel_history = vel_history
        self.dt = dt
        self.lambda2_history = lambda2_history
        self.edges_history = edges_history
        self.damping = damping
        self.engine = engine 
        self.n_robots, self.n_steps, self.dim = history.shape
        self.robot_colors = plt.cm.tab10(np.linspace(0, 1, self.n_robots))
        self.meeting_point = np.mean(self.history[:, 0, :], axis=0)
        self.target_fps = target_fps
        self.interval = int(1000 / self.target_fps)
        self.stride = max(1, int((1.0 / self.target_fps) / self.dt))        
        self.n_frames_anim = self.n_steps // self.stride

    def _get_dynamic_info_str(self, lambda2):
        if self.engine is None:
            return "Dados da Engine ausentes"
            
        kp = self.engine.protocol.kp
        kd = self.engine.protocol.kd
        ki = self.engine.protocol.ki
        
        if lambda2 == 0:
            return (
                r"$\mathbf{Status:}$ Rede Desconectada ($\lambda_2 = 0$)" + "\n"
                r"Consenso impossível neste frame (Teorema 3)."
            )
            
        if ki == 0:
            # Avaliação de 2ª Ordem baseada no Teorema 8 de Olfati-Saber (2007)
            info_str = (
                r"$\mathbf{Din\^amica:}$ $\ddot{x} + k_d L \dot{x} + k_p L x = 0$ (Eq 4.11)" + "\n"
                r"$\mathbf{Amortecimento:}$ $\zeta_2 = \frac{k_d \sqrt{\lambda_2}}{2\sqrt{k_p}}$ (via Teorema 8)" + "\n\n"
                r"$\mathbf{Par\^ametros:}$" + f" $k_p={kp:.2f}, k_d={kd:.2f}$\n"
            )
            
            if kp <= 0:
                return info_str + r"$\rightarrow$ Erro Crítico: $k_p \le 0$"
                
            zeta = (kd * np.sqrt(lambda2)) / (2.0 * np.sqrt(kp))
            info_str += f"$\\zeta_2 = {zeta:.3f}$\n"
            
            if zeta == 0:
                info_str += r"$\rightarrow$ Marginalmente Estável (Oscilação Pura)"
            elif np.isclose(zeta, 1.0, atol=1e-3): 
                info_str += r"$\rightarrow$ Criticamente Amortecido (Estável)"
            elif zeta < 1:
                info_str += r"$\rightarrow$ Subamortecido (Estável)"
            else:
                info_str += r"$\rightarrow$ Sobreamortecido (Estável)"
                
            return info_str
            
        else:
            # Avaliação de 3ª Ordem (PID) via Routh-Hurwitz e Controle de 3ª Ordem
            info_str = (
                r"$\mathbf{Din\^amica:}$ $\dddot{x} + k_d L \ddot{x} + k_p L \dot{x} + k_i L x = 0$ (Eq 6.8)" + "\n"
                r"$\mathbf{Estabilidade (R-H):}$ $k_d k_p \lambda_2 > k_i$" + "\n\n"
                r"$\mathbf{Par\^ametros:}$" + f" $k_p={kp:.2f}, k_d={kd:.2f}, k_i={ki:.2f}$\n"
            )
            
            rh_val = kd * kp * lambda2
            info_str += f"R-H Eval: ${rh_val:.2f} > {ki:.2f}$\n"
            
            if rh_val > ki:
                info_str += r"$\rightarrow$ Estável (3ª Ordem)"
            else:
                info_str += r"$\rightarrow$ Instável (Viola Routh-Hurwitz)"
                
            return info_str

    def _create_matrices_window(self):
        if not self.engine:
            return
            
        _, L = self.engine.topology.compute_graph_matrices(self.history[:, -1, :])
        n = self.n_robots
        kp = self.engine.protocol.kp
        ki = self.engine.protocol.ki
        kd = self.engine.protocol.kd
        
        # Matrizes Locais (Dinâmica do Agente Isolado)
        A_local = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        B_local = np.array([[0], [0], [1]])
        K_local = np.array([[ki, kp, kd]])
        
        # Dinâmica Global em Malha Aberta
        I_n = np.eye(n)
        A_open = np.kron(I_n, A_local)
        B_open = np.kron(I_n, B_local)
        
        # Dinâmica Coletiva (Malha Fechada)
        K_global = np.kron(L, K_local)
        M_collective = A_open - B_open @ K_global
        
        # Calcula o tamanho da janela dinamicamente com base no número de robôs
        largura = min(16, 8 + n * 1.2)
        altura = min(10, 4 + n * 0.8)
        fig_mat, axes = plt.subplots(1, 2, figsize=(largura, altura))
        
        if hasattr(fig_mat.canvas.manager, 'set_window_title'):
            fig_mat.canvas.manager.set_window_title("Dinâmica Coletiva MIMO")
            
        for ax in axes:
            ax.axis('off')
        
        opt = {'precision': 2, 'suppress_small': True, 'max_line_width': 150, 'threshold': np.inf}
        
        texto_col1 = (
            "Dinamica Local do Agente 'i' (Eq. 38):\n"
            "  x_dot_i = A * x_i + B * u_i\n\n"
            "Matrizes Locais do Agente:\n"
            f"A =\n{np.array2string(A_local, **opt)}\n"
            f"B =\n{np.array2string(B_local, **opt)}\n"
            f"K = {np.array2string(K_local, **opt)}\n\n"
            "Dinamica Global em Malha Aberta:\n"
            "  X_dot = (I_n \u2297 A) * X  +  (I_n \u2297 B) * U\n\n"
            "Protocolo de Consenso:\n"
            "  U = -(L \u2297 K) * X\n\n"
            "Dinamica Coletiva Global (Eq. 43 / Eq. 4.29):\n"
            "  X_dot = [ (I_n \u2297 A) - (L \u2297 (B * K)) ] * X = M * X\n"
            "\n\n==========================================================\n\n"
            f"Matriz Laplaciana L ({n}x{n}):\n{np.array2string(L, **opt)}\n\n"
            f"Ganhos Relativos Globais (L \u2297 K) {K_global.shape}:\n{np.array2string(K_global, **opt)}"
        )
        
        texto_col2 = (
            f"Matriz Malha Aberta Global (I_n \u2297 A)  {A_open.shape}:\n{np.array2string(A_open, **opt)}\n\n"
            "\n\n==========================================================\n\n"
            f"Matriz de Dinamica Coletiva M {M_collective.shape}:\n{np.array2string(M_collective, **opt)}"
        )
        
        # Desenha Coluna Esquerda
        axes[0].text(0.02, 0.98, texto_col1, transform=axes[0].transAxes, fontsize=9, fontfamily='monospace',
                     verticalalignment='top', bbox=dict(boxstyle='square', facecolor='white', edgecolor='lightgray'))
        
        # Desenha Coluna Direita
        axes[1].text(0.02, 0.98, texto_col2, transform=axes[1].transAxes, fontsize=9, fontfamily='monospace',
                     verticalalignment='top', bbox=dict(boxstyle='square', facecolor='whitesmoke', edgecolor='lightgray'))
        
        plt.tight_layout()

    def plot_analysis(self):
        time_axis = np.linspace(0, self.n_steps * self.dt, self.n_steps)
        fig = plt.figure(figsize=(12, 10))
        if hasattr(fig.canvas.manager, 'set_window_title'):
            fig.canvas.manager.set_window_title("Análise de Estabilidade e Desempenho")

        ax_poles_ind = plt.subplot2grid((3, 2), (0, 0))
        ax_poles_glob = plt.subplot2grid((3, 2), (0, 1))
        
        if self.engine:
            # 1. Polos Globais (Malha Fechada)
            _, L = self.engine.topology.compute_graph_matrices(self.history[:, -1, :])
            A_global = self.engine.get_global_state_space_matrix(L)
            polos_globais = np.linalg.eigvals(A_global)
            
            # 2. Polos Individuais (Malha Aberta -> Forçando L = 0)
            L_zeros = np.zeros_like(L)
            A_ind = self.engine.get_global_state_space_matrix(L_zeros)
            polos_ind = np.linalg.eigvals(A_ind)

            # 3. Cálculos de quantidade para exibição didática
            total_polos = len(polos_globais)
            polos_por_agente = total_polos // self.n_robots

            # Plot Individuais
            ax_poles_ind.scatter(np.real(polos_ind), np.imag(polos_ind), marker='x', color='red', s=40, label='Polos')
            ax_poles_ind.axvline(0, color='black', lw=1)
            ax_poles_ind.axhline(0, color='black', lw=1)
            ax_poles_ind.set_title(f"Polos Individuais (Malha Aberta) (Eq 3.2)\nTotal: {total_polos} polos ({polos_por_agente} por robô)")
            ax_poles_ind.set_xlabel("Eixo Real")
            ax_poles_ind.set_ylabel("Eixo Imaginário")
            ax_poles_ind.grid(True)
            ax_poles_ind.legend()

            # Plot Globais
            ax_poles_glob.scatter(np.real(polos_globais), np.imag(polos_globais), marker='x', color='red', s=40, label='Polos')
            ax_poles_glob.axvline(0, color='black', lw=1)
            ax_poles_glob.axhline(0, color='black', lw=1)
            ax_poles_glob.set_title(f"Polos Globais (Malha Fechada) (Seção IV-B)\nTotal: {total_polos} polos ({polos_por_agente} por robô)")
            ax_poles_glob.set_xlabel("Eixo Real")
            ax_poles_glob.set_ylabel("Eixo Imaginário")
            ax_poles_glob.grid(True)
            ax_poles_glob.legend()

        ax_lambda = plt.subplot2grid((3, 2), (1, 0), colspan=2)
        if self.lambda2_history is not None:
            ax_lambda.plot(time_axis, self.lambda2_history, color='black', lw=2)
            ax_lambda.set_title(r"Conectividade Algébrica ($\lambda_2$) (Eq 12)")
            ax_lambda.set_xlabel("Tempo (s)")
            ax_lambda.grid(True)

        ax_dist = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        for i in range(self.n_robots):
            if self.dim == 1:
                dists = [self.history[i, t, 0] - self.meeting_point[0] for t in range(self.n_steps)]
                ylabel_str = "Erro de Posição (x - x_encontro)"
            else:
                dists = [np.linalg.norm(self.history[i, t] - self.meeting_point) for t in range(self.n_steps)]
                ylabel_str = "Distância Euclidiana até o Centro"
                
            ax_dist.plot(time_axis, dists, color=self.robot_colors[i], label=f'Robô {i}', alpha=0.8)
        
        ax_dist.set_title("Distância / Erro de Posição (Disagreement Function Eq 10)")
        ax_dist.set_xlabel("Tempo (s)")
        ax_dist.set_ylabel(ylabel_str)
        ax_dist.grid(True)
        ax_dist.axhline(0, color='black', lw=1.5, linestyle='--')
        
        ax_dist.legend(loc='upper right', ncol=2, fontsize='small')

        plt.tight_layout()
        
        self._create_matrices_window()
        plt.show()

    def animate(self):
        if self.dim == 1:
            self._animate_1d()
        elif self.dim == 2:
            self._animate_2d()
        elif self.dim == 3:
            self._animate_3d()
        else:
            print(f"Amount of dimensions {self.dim} > 3. Reducing to 3D using PCA.")
            self._animate_pca()

    def _animate_1d(self):
        fig, ax = plt.subplots(figsize=(8, 3))
        all_x = self.history[:, :, 0]
        ax.set_xlim(np.min(all_x) - 1, np.max(all_x) + 1)
        ax.set_ylim(-1, 1)
        ax.set_title("Rendezvous 1D")
        ax.set_xlabel("Position")
        ax.set_ylabel("(y=0)")
        ax.grid(True)

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        lambda_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, fontsize=10,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        eq_text = ax.text(0.02, 0.76, '', transform=ax.transAxes, fontsize=10,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        init_x = self.history[:, 0, 0]
        init_y = np.zeros_like(init_x)
        scatters = ax.scatter(init_x, init_y, c=self.robot_colors, s=50)
        lines = [ax.plot([], [], '--', alpha=0.3, color=self.robot_colors[i])[0] for i in range(self.n_robots)]

        meeting_x = self.meeting_point[0]
        ax.axvline(x=meeting_x, color='red', linestyle='--', linewidth=2, label='Rendezvous point')
        ax.scatter(meeting_x, 0, marker='x', color='red', s=100)

        handles = [Patch(color=self.robot_colors[i], label=f'Robot {i}') for i in range(self.n_robots)]
        handles.append(plt.Line2D([0], [0], color='red', linestyle='--', label='Rendezvous point'))
        ax.legend(handles=handles)

        def update(anim_frame):
            frame = anim_frame * self.stride

            for i in range(self.n_robots):
                x_vals = self.history[i, :frame, 0]
                y_vals = np.zeros_like(x_vals)
                lines[i].set_data(x_vals, y_vals)

            curr_pos = self.history[:, frame, 0]
            y_curr = np.zeros_like(curr_pos)
            scatters.set_offsets(np.c_[curr_pos, y_curr])
            
            time_text.set_text(f't = {frame * self.dt:.2f} s')
            
            if self.lambda2_history is not None:
                lambda2 = self.lambda2_history[frame]
                lambda_text.set_text(f'λ₂ = {lambda2:.4f}')
                
                info_str = self._get_dynamic_info_str(lambda2)
                eq_text.set_text(info_str)
            else:
                lambda_text.set_text('λ₂ = N/A')
                eq_text.set_text("Dados insuficientes")
            
            return lines + [scatters, time_text, lambda_text, eq_text]

        ani = FuncAnimation(fig, update, frames=self.n_frames_anim, interval=self.interval, blit=True)
        plt.show()

    def _animate_2d(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        all_x = self.history[:, :, 0]
        all_y = self.history[:, :, 1]
        ax.set_xlim(np.min(all_x) - 1, np.max(all_x) + 1)
        ax.set_ylim(np.min(all_y) - 1, np.max(all_y) + 1)
        ax.set_title("Rendezvous 2D")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        lambda_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        eq_text = ax.text(0.02, 0.80, '', transform=ax.transAxes, fontsize=10,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        init_x = self.history[:, 0, 0]
        init_y = self.history[:, 0, 1]
        scatters = ax.scatter(init_x, init_y, c=self.robot_colors, s=50)

        trail_lines = [ax.plot([], [], '--', alpha=0.3, color=self.robot_colors[i])[0] for i in range(self.n_robots)]

        edge_lines = []
        edge_line_dict = {}
        if self.edges_history is not None:
            for i in range(self.n_robots):
                for j in range(i+1, self.n_robots):
                    line, = ax.plot([], [], '-', color='gray', alpha=0.6, linewidth=1.5)
                    edge_line_dict[(i, j)] = line
                    edge_lines.append(line)

        meeting_x, meeting_y = self.meeting_point[0], self.meeting_point[1]
        ax.scatter(meeting_x, meeting_y, marker='x', color='red', s=100, label='Rendezvous point')

        handles = [Patch(color=self.robot_colors[i], label=f'Robot {i}') for i in range(self.n_robots)]
        handles.append(plt.Line2D([0], [0], marker='x', color='red', linestyle='None', label='Rendezvous point'))
        ax.legend(handles=handles)

        def update(anim_frame):
            frame = anim_frame * self.stride

            for i in range(self.n_robots):
                trail_lines[i].set_data(self.history[i, :frame, 0], self.history[i, :frame, 1])

            curr_pos = self.history[:, frame, :]
            scatters.set_offsets(curr_pos)

            if self.edges_history is not None:
                for line in edge_line_dict.values():
                    line.set_data([], [])
                for u, v in self.edges_history[frame]:
                    x = [self.history[u, frame, 0], self.history[v, frame, 0]]
                    y = [self.history[u, frame, 1], self.history[v, frame, 1]]
                    key = (u, v) if u < v else (v, u)
                    if key in edge_line_dict:
                        edge_line_dict[key].set_data(x, y)

            time_text.set_text(f't = {frame * self.dt:.2f} s')
            
            if self.lambda2_history is not None:
                lambda2 = self.lambda2_history[frame]
                lambda_text.set_text(f'λ₂ = {lambda2:.4f}')
                
                info_str = self._get_dynamic_info_str(lambda2)
                eq_text.set_text(info_str)
            else:
                lambda_text.set_text('λ₂ = N/A')
                eq_text.set_text("Dados insuficientes")

            artists = [scatters, time_text, lambda_text, eq_text] + trail_lines + edge_lines
            return artists

        ani = FuncAnimation(fig, update, frames=self.n_frames_anim, interval=self.interval, blit=True)
        plt.show()
    
    def _animate_3d(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        all_x = self.history[:, :, 0]
        all_y = self.history[:, :, 1]
        all_z = self.history[:, :, 2]
        ax.set_xlim(np.min(all_x) - 1, np.max(all_x) + 1)
        ax.set_ylim(np.min(all_y) - 1, np.max(all_y) + 1)
        ax.set_zlim(np.min(all_z) - 1, np.max(all_z) + 1)
        ax.set_title("Rendezvous 3D")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        lambda_text = ax.text2D(0.02, 0.88, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        eq_text = ax.text2D(0.02, 0.80, '', transform=ax.transAxes, fontsize=10,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        scatters = ax.scatter(self.history[:, 0, 0], self.history[:, 0, 1], self.history[:, 0, 2],
                            c=self.robot_colors, s=50)
        lines = [ax.plot([], [], [], '--', alpha=0.3, color=self.robot_colors[i])[0] for i in range(self.n_robots)]
        
        edge_lines_3d = []
        edge_line_dict_3d = {}
        if self.edges_history is not None:
            for i in range(self.n_robots):
                for j in range(i+1, self.n_robots):
                    line, = ax.plot([], [], [], '-', color='gray', alpha=0.6, linewidth=1.5)
                    edge_line_dict_3d[(i, j)] = line
                    edge_lines_3d.append(line)

        meeting_x, meeting_y, meeting_z = self.meeting_point
        ax.scatter(meeting_x, meeting_y, meeting_z, marker='x', color='red', s=100, label='Rendezvous point')

        handles = [Patch(color=self.robot_colors[i], label=f'Robot {i}') for i in range(self.n_robots)]
        handles.append(plt.Line2D([0], [0], marker='x', color='red', linestyle='None', label='Rendezvous point'))
        ax.legend(handles=handles)

        def update(anim_frame):
            frame = anim_frame * self.stride

            for i in range(self.n_robots):
                x_vals = self.history[i, :frame, 0]
                y_vals = self.history[i, :frame, 1]
                z_vals = self.history[i, :frame, 2]
                lines[i].set_data(x_vals, y_vals)
                lines[i].set_3d_properties(z_vals)

            curr_pos = self.history[:, frame, :]
            scatters._offsets3d = (curr_pos[:, 0], curr_pos[:, 1], curr_pos[:, 2])
            
            if self.edges_history is not None:
                for line in edge_line_dict_3d.values():
                    line.set_data([], [])
                    line.set_3d_properties([])
                for u, v in self.edges_history[frame]:
                    x = [self.history[u, frame, 0], self.history[v, frame, 0]]
                    y = [self.history[u, frame, 1], self.history[v, frame, 1]]
                    z = [self.history[u, frame, 2], self.history[v, frame, 2]]
                    key = (u, v) if u < v else (v, u)
                    if key in edge_line_dict_3d:
                        edge_line_dict_3d[key].set_data(x, y)
                        edge_line_dict_3d[key].set_3d_properties(z)
                
            time_text.set_text(f't = {frame * self.dt:.2f} s')
            
            if self.lambda2_history is not None:
                lambda2 = self.lambda2_history[frame]
                lambda_text.set_text(f'λ₂ = {lambda2:.4f}')
                
                info_str = self._get_dynamic_info_str(lambda2)
                eq_text.set_text(info_str)
            else:
                lambda_text.set_text('λ₂ = N/A')
                eq_text.set_text("Dados insuficientes")

            return lines + [scatters, time_text, lambda_text, eq_text]

        ani = FuncAnimation(fig, update, frames=self.n_frames_anim, interval=self.interval, blit=False)
        plt.show()
    
    def _animate_pca(self):
        flat_data = self.history.reshape(-1, self.dim)
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(flat_data)
        reduced_3d = reduced.reshape(self.n_robots, self.n_steps, 3)

        meeting_original = self.meeting_point.reshape(1, -1)
        meeting_pca = pca.transform(meeting_original)[0]

        original_history = self.history
        original_dim = self.dim
        original_meeting = self.meeting_point
        self.history = reduced_3d
        self.dim = 3
        self.meeting_point = meeting_pca

        print("Animating with PCA projection to 3D. Variance explanation:", pca.explained_variance_ratio_)
        self._animate_3d() 

        self.history = original_history
        self.dim = original_dim
        self.meeting_point = original_meeting