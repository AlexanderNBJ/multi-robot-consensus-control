import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

class Visualizer:
    def __init__(self, history, dt, lambda2_history=None, edges_history = None):
        self.history = history
        self.dt = dt
        self.lambda2_history = lambda2_history
        self.edges_history = edges_history
        self.n_robots, self.n_steps, self.dim = history.shape
        self.robot_colors = plt.cm.tab10(np.linspace(0, 1, self.n_robots))
        self.meeting_point = np.mean(self.history[:, 0, :], axis=0)

    def plot_analysis(self):
        steps = self.n_steps
        time_axis = np.linspace(0, steps * self.dt, steps)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Gráfico das distâncias até o ponto de encontro
        for i in range(self.n_robots):
            distances = [np.linalg.norm(self.history[i, t] - self.meeting_point) for t in range(steps)]
            ax1.plot(time_axis, distances, color=self.robot_colors[i], label=f'Robot {i}')
        ax1.set_ylabel('Distance')
        ax1.set_title('Distance of each robot to the Rendezvous Point')
        ax1.grid(True)
        ax1.legend()

        # Gráfico da conectividade algébrica
        if self.lambda2_history is not None:
            ax2.plot(time_axis, self.lambda2_history, 'k-', linewidth=2)
            ax2.set_ylabel('Algebraic Connectivity λ₂')
            ax2.set_xlabel('Time (s)')
            ax2.grid(True)
        else:
            ax2.set_visible(False)  # esconde se não houver dados

        plt.tight_layout()
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

        def update(frame):
            for i in range(self.n_robots):
                x_vals = self.history[i, :frame, 0]
                y_vals = np.zeros_like(x_vals)
                lines[i].set_data(x_vals, y_vals)

            curr_pos = self.history[:, frame, 0]
            y_curr = np.zeros_like(curr_pos)
            scatters.set_offsets(np.c_[curr_pos, y_curr])
            time_text.set_text(f't = {frame * self.dt:.2f} s')
            
            if self.lambda2_history is not None:
                lambda_text.set_text(f'λ₂ = {self.lambda2_history[frame]:.4f}')
            else:
                lambda_text.set_text('λ₂ = N/A')
            
            return lines + [scatters, time_text, lambda_text]

        ani = FuncAnimation(fig, update, frames=self.n_steps, interval=20, blit=True)
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

        # Scatter inicial (posições do frame 0)
        init_x = self.history[:, 0, 0]
        init_y = self.history[:, 0, 1]
        scatters = ax.scatter(init_x, init_y, c=self.robot_colors, s=50)

        # Trilhas (histórico)
        trail_lines = [ax.plot([], [], '--', alpha=0.3, color=self.robot_colors[i])[0] for i in range(self.n_robots)]

        # Linhas das arestas (inicialmente vazias)
        edge_lines = []
        edge_line_dict = {}
        if self.edges_history is not None:
            for i in range(self.n_robots):
                for j in range(i+1, self.n_robots):
                    line, = ax.plot([], [], '-', color='gray', alpha=0.6, linewidth=1.5)
                    edge_line_dict[(i, j)] = line
                    edge_lines.append(line)

        # Ponto de encontro
        meeting_x, meeting_y = self.meeting_point[0], self.meeting_point[1]
        ax.scatter(meeting_x, meeting_y, marker='x', color='red', s=100, label='Rendezvous point')

        handles = [Patch(color=self.robot_colors[i], label=f'Robot {i}') for i in range(self.n_robots)]
        handles.append(plt.Line2D([0], [0], marker='x', color='red', linestyle='None', label='Rendezvous point'))
        ax.legend(handles=handles)

        def update(frame):
            # 1. Atualiza as trilhas
            for i in range(self.n_robots):
                trail_lines[i].set_data(self.history[i, :frame, 0], self.history[i, :frame, 1])

            # 2. Atualiza posições atuais (scatter)
            curr_pos = self.history[:, frame, :]
            scatters.set_offsets(curr_pos)

            # 3. Atualiza as arestas (se existirem)
            if self.edges_history is not None:
                # Primeiro, esconde todas as arestas
                for line in edge_line_dict.values():
                    line.set_data([], [])
                # Agora desenha as arestas ativas neste frame
                for u, v in self.edges_history[frame]:
                    x = [self.history[u, frame, 0], self.history[v, frame, 0]]
                    y = [self.history[u, frame, 1], self.history[v, frame, 1]]
                    # Garante que a chave seja ordenada (menor, maior)
                    key = (u, v) if u < v else (v, u)
                    if key in edge_line_dict:
                        edge_line_dict[key].set_data(x, y)

            # 4. Textos
            time_text.set_text(f't = {frame * self.dt:.2f} s')
            if self.lambda2_history is not None:
                lambda_text.set_text(f'λ₂ = {self.lambda2_history[frame]:.4f}')
            else:
                lambda_text.set_text('λ₂ = N/A')

            # Retorna todos os artistas que mudaram (para blit=True)
            artists = [scatters, time_text, lambda_text] + trail_lines + edge_lines
            return artists

        ani = FuncAnimation(fig, update, frames=self.n_steps, interval=20, blit=True)
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

        def update(frame):
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
                lambda_text.set_text(f'λ₂ = {self.lambda2_history[frame]:.4f}')
            else:
                lambda_text.set_text('λ₂ = N/A')
            return lines + [scatters, time_text, lambda_text]

        ani = FuncAnimation(fig, update, frames=self.n_steps, interval=20, blit=False)
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