import argparse
import json
import logging
import sys
import numpy as np
import matplotlib
import math
matplotlib.use('TkAgg')
from Visualizer import Visualizer
from Engine import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def parse_vector_parameter(data_val, n_robots, dim, default_val=0.0):
    if data_val is None:
        return np.full((n_robots, dim), default_val)

    if isinstance(data_val, list):
        data_array = np.array(data_val, dtype=float)
        
        if data_array.ndim == 1:
            if len(data_array) != dim:
                raise ValueError(f"Tamanho do vetor {len(data_array)} não bate com a dimensão {dim}.")
            return np.tile(data_array, (n_robots, 1))
            
        elif data_array.ndim == 2:
            if data_array.shape != (n_robots, dim):
                raise ValueError(f"Esperada matriz ({n_robots}, {dim}), mas recebeu {data_array.shape}.")
            return data_array

    if isinstance(data_val, dict):
        matrix = np.full((n_robots, dim), default_val)
        for key, val in data_val.items():
            idx = int(key)
            if 0 <= idx < n_robots:
                vec = np.array(val, dtype=float)
                if len(vec) == dim:
                    matrix[idx] = vec
                else:
                    raise ValueError(f"Vetor no índice {idx} não bate com a dimensão {dim}.")
        return matrix

    raise ValueError(f"Formato não suportado para o parâmetro: {type(data_val)}")

def generate_preset_positions(preset: str, params: dict) -> list:
    n = params.get('n_robots', 5)
    dim = params.get('dim', 2)
    
    if n <= 0 or dim <= 0:
        raise ValueError(f"Dimensão ({dim}) e número de robôs ({n}) devem ser positivos.")

    raw_center = params.get('center', [0.0] * dim)
    center = np.array(raw_center, dtype=float)[:dim]
    if len(center) < dim:
        center = np.pad(center, (0, dim - len(center)), 'constant')

    preset = preset.lower()
    
    if preset in ['random', 'random_uniform']:
        box_size = params.get('box_size', 10.0)
        low = center - (box_size / 2)
        high = center + (box_size / 2)
        return np.random.uniform(low, high, size=(n, dim)).tolist()

    elif preset in ['gaussian', 'random_gaussian']:
        std_dev = params.get('std_dev', 2.0)
        return (center + np.random.normal(scale=std_dev, size=(n, dim))).tolist()

    elif preset == 'hypersphere_volume':
        radius = params.get('radius', 5.0)
        points = np.random.normal(size=(n, dim))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        surface = points / norms
        
        u = np.random.uniform(size=(n, 1))
        scale = radius * (u ** (1.0 / dim))
        return (center + surface * scale).tolist()

    elif preset == 'hypersphere_surface':
        radius = params.get('radius', 5.0)
        points = np.random.normal(size=(n, dim))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (center + (points / norms) * radius).tolist()

    elif preset == 'line':
        length = params.get('length', 10.0)
        raw_dir = params.get('direction', [1.0] + [0.0]*(dim-1))
        direction = np.array(raw_dir, dtype=float)[:dim]
        if len(direction) < dim:
            direction = np.pad(direction, (0, dim - len(direction)), 'constant')
            
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 0 else np.zeros(dim)
        
        t = np.linspace(-length/2, length/2, n)
        return (center + np.outer(t, direction)).tolist()

    elif preset in ['regular_polygon', 'ring']:
        radius = params.get('radius', 5.0)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.zeros((n, dim))
        pts[:, 0] = np.cos(angles) * radius
        if dim > 1:
            pts[:, 1] = np.sin(angles) * radius
        return (center + pts).tolist()

    elif preset == 'spiral':
        turns = params.get('turns', 3.0)
        spacing = params.get('spacing', 1.0)
        t = np.linspace(0.1, turns * 2 * np.pi, n)
        r = spacing * t / (2 * np.pi)
        
        pts = np.zeros((n, dim))
        pts[:, 0] = r * np.cos(t)
        if dim > 1:
            pts[:, 1] = r * np.sin(t)
        if dim > 2:
            height = params.get('height', 10.0)
            pts[:, 2] = np.linspace(-height/2, height/2, n)
        return (center + pts).tolist()

    elif preset == 'grid':
        spacing = params.get('spacing', 2.0)
        m = math.ceil(n ** (1.0 / dim))
        
        if m ** dim > 1_000_000:
            logging.warning(f"Grid {dim}D para {n} robôs causaria estouro de memória. Usando random_uniform como fallback.")
            return generate_preset_positions('random_uniform', params)

        axes = [np.arange(m) * spacing for _ in range(dim)]
        mesh = np.meshgrid(*axes, indexing='ij')
        points = np.vstack([x.flatten() for x in mesh]).T
        
        points = points - np.mean(points, axis=0) + center
        return points[:n].tolist()

    else:
        raise ValueError(f"Preset desconhecido: '{preset}'. Verifique o config.json.")

def build_simulator_from_dict(data: dict) -> 'SimulatorEngine':
    if 'preset' in data:
        initial_positions = generate_preset_positions(data['preset'], data.get('preset_params', {}))
    else:
        initial_positions = data.get('initial_robot_positions', [])

    n_robots = len(initial_positions)
    dim = len(initial_positions[0]) if n_robots > 0 else 2

    import math
    raw_radius = data.get('rendezvous_radius', 1.0)
    radius = math.inf if str(raw_radius).upper() == 'INFINITY' else float(raw_radius)

    topology = NetworkTopology(
        use_proximity=data.get('use_proximity', True),
        rendezvous_radius=radius,
        fixed_edges=data.get('fixed_edges', []),
        prohibited_edges=data.get('prohibited_edges', [])
    )
    
    bias = parse_vector_parameter(data.get('bias'), n_robots, dim)
    
    abs_offsets = parse_vector_parameter(data.get('formation_offsets'), n_robots, dim)
    rel_offsets = parse_vector_parameter(data.get('relative_offsets'), n_robots, dim, default_val=None)

    controller = ConsensusController(
        offsets=abs_offsets if rel_offsets is None else None,
        bias=bias,
        angular_velocity=data.get('angular_velocity', 0.0),
        rotation_center_index=data.get('rotation_center_index'),
        relative_offsets=rel_offsets,
        damping=data.get('damping', 1.0)
    )
    
    dynamics_order = data.get('dynamics_order', 1)
    
    return SimulatorEngine(initial_positions, topology, controller, dynamics_order=dynamics_order)

def main():
    parser = argparse.ArgumentParser(description="Simulador de Rendezvous de Robôs Multi-Agentes")
    parser.add_argument(
        '-c', '--config', 
        type=str, 
        required=True, 
        help="Caminho para o arquivo JSON de configuração"
    )
    parser.add_argument('--dt', type=float, default=0.01, help="Passo de tempo da simulação")
    parser.add_argument('--steps', type=int, default=1000, help="Número de passos da simulação")
    parser.add_argument('--no-plot', action='store_true', help="Roda a simulação sem abrir os gráficos")

    args = parser.parse_args()

    try:
        with open(args.config, 'r') as file:
            config_data = json.load(file)
            logging.info(f"Configuração carregada com sucesso: {args.config}")
    except Exception as e:
        logging.error(f"Falha ao ler o arquivo de configuração: {e}")
        sys.exit(1)

    try:
        sim = build_simulator_from_dict(config_data)
        logging.info(f"Iniciando simulação com {sim.n_robots} robôs em {sim.dim}D.")
        logging.info(f"Integrando {args.steps} passos com dt={args.dt} usando dinâmica de ordem {sim.dynamics_order}...")
        
        history, lambda2_history, edges_history = sim.run(dt=args.dt, steps=args.steps)
        logging.info("Simulação concluída.")

    except Exception as e:
        logging.error(f"Erro durante a execução da simulação: {e}")
        sys.exit(1)

    if not args.no_plot:
        logging.info("Iniciando visualizador...")
        viz = Visualizer(history, args.dt, lambda2_history, edges_history)
        viz.animate()
        viz.plot_analysis()
    else:
        logging.info("Visualização ignorada (--no-plot ativado).")

if __name__ == '__main__':
    main()