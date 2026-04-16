"""
Modulo do ambiente Gridworld com perigos.

Implementa um MDP discreto e finito com:
- Espacos de estados: coordenadas (row, col) mapeadas para indices lineares
- Acoes: {0: UP, 1: RIGHT, 2: DOWN, 3: LEFT}
- Transicoes estocasticas (vento/piso escorregadio)
- Recompensas: objetivo (+1), armadilhas (-1), custo de movimento (-0.04)
"""

import numpy as np


# Tipos de celula
EMPTY = 0
WALL = 1
TRAP = 2
GOAL = 3
START = 4

# Acoes
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]
NUM_ACTIONS = 4

# Deltas de movimento para cada acao: (delta_row, delta_col)
ACTION_DELTAS = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}


class GridworldEnv:
    """
    Ambiente Gridworld com perigos e transicoes estocasticas.

    Parametros
    ----------
    grid_layout : list of list of int
        Matriz representando o layout da grade. Cada celula eh um dos tipos:
        0=EMPTY, 1=WALL, 2=TRAP, 3=GOAL, 4=START
    slip_prob : float
        Probabilidade de escorregar para uma direcao perpendicular (total).
        A acao pretendida tem prob (1 - slip_prob), cada perpendicular tem slip_prob/2.
    reward_goal : float
        Recompensa ao atingir o objetivo.
    reward_trap : float
        Penalidade ao cair em armadilha.
    reward_step : float
        Custo de movimento por passo (living penalty).
    seed : int or None
        Semente aleatoria para reprodutibilidade.
    """

    def __init__(
        self,
        grid_layout=None,
        slip_prob=0.2,
        reward_goal=1.0,
        reward_trap=-1.0,
        reward_step=-0.04,
        seed=42,
    ):
        if grid_layout is None:
            grid_layout = self._default_layout()

        self.grid = np.array(grid_layout, dtype=int)
        self.rows, self.cols = self.grid.shape
        self.slip_prob = slip_prob
        self.reward_goal = reward_goal
        self.reward_trap = reward_trap
        self.reward_step = reward_step
        self.rng = np.random.default_rng(seed)

        # Encontrar posicao inicial e posicoes terminais
        start_positions = list(zip(*np.where(self.grid == START)))
        if not start_positions:
            raise ValueError("O grid deve conter pelo menos uma celula START (4).")
        self.start_pos = start_positions[0]

        # Construir espaco de estados validos
        self.states = []
        self.state_to_idx = {}
        idx = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] != WALL:
                    self.states.append((r, c))
                    self.state_to_idx[(r, c)] = idx
                    idx += 1
        self.num_states = len(self.states)

        # Construir matriz de transicao P[s][a] = [(prob, s', reward, done), ...]
        self.P = self._build_transition_model()

        # Estado atual do agente
        self.agent_pos = None

    def _default_layout(self):
        """
        Layout padrao 4x4 (classico do Frozen Lake / Russell & Norvig):

        S  .  .  .
        .  W  .  T
        .  .  .  T
        T  .  .  G

        S=start, W=wall, T=trap, G=goal
        """
        return [
            [START, EMPTY, EMPTY, EMPTY],
            [EMPTY, WALL, EMPTY, TRAP],
            [EMPTY, EMPTY, EMPTY, TRAP],
            [TRAP, EMPTY, EMPTY, GOAL],
        ]

    def _is_valid(self, r, c):
        """Verifica se a posicao esta dentro da grade e nao eh parede."""
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] != WALL

    def _get_next_pos(self, r, c, action):
        """Retorna a proxima posicao apos executar a acao. Se bater na parede, fica no lugar."""
        dr, dc = ACTION_DELTAS[action]
        nr, nc = r + dr, c + dc
        if self._is_valid(nr, nc):
            return (nr, nc)
        return (r, c)

    def _get_reward(self, pos):
        """Retorna a recompensa para a posicao destino."""
        cell = self.grid[pos[0], pos[1]]
        if cell == GOAL:
            return self.reward_goal
        elif cell == TRAP:
            return self.reward_trap
        else:
            return self.reward_step

    def _is_terminal(self, pos):
        """Verifica se a posicao eh terminal (goal ou trap)."""
        cell = self.grid[pos[0], pos[1]]
        return cell == GOAL or cell == TRAP

    def _build_transition_model(self):
        """
        Constroi o modelo completo de transicao para Value Iteration.

        P[s_idx][a] = lista de (probabilidade, s'_idx, recompensa, terminal)
        """
        P = {}
        for s_idx, (r, c) in enumerate(self.states):
            P[s_idx] = {}
            if self._is_terminal((r, c)):
                # Estados terminais: qualquer acao leva a si mesmo com recompensa 0
                for a in range(NUM_ACTIONS):
                    P[s_idx][a] = [(1.0, s_idx, 0.0, True)]
                continue

            for a in range(NUM_ACTIONS):
                transitions = []

                # Acoes possiveis: pretendida e perpendiculares
                if self.slip_prob > 0:
                    intended_prob = 1.0 - self.slip_prob
                    perp_prob = self.slip_prob / 2.0

                    # Acao pretendida
                    intended_actions = [(intended_prob, a)]

                    # Acoes perpendiculares
                    if a == UP or a == DOWN:
                        perp_actions = [(perp_prob, LEFT), (perp_prob, RIGHT)]
                    else:
                        perp_actions = [(perp_prob, UP), (perp_prob, DOWN)]

                    all_actions = intended_actions + perp_actions
                else:
                    all_actions = [(1.0, a)]

                # Agregar transicoes por estado destino
                trans_dict = {}
                for prob, act in all_actions:
                    next_pos = self._get_next_pos(r, c, act)
                    next_idx = self.state_to_idx[next_pos]
                    reward = self._get_reward(next_pos)
                    done = self._is_terminal(next_pos)

                    if next_idx in trans_dict:
                        trans_dict[next_idx] = (
                            trans_dict[next_idx][0] + prob,
                            next_idx,
                            reward,
                            done,
                        )
                    else:
                        trans_dict[next_idx] = (prob, next_idx, reward, done)

                P[s_idx][a] = list(trans_dict.values())

        return P

    def reset(self, seed=None):
        """Reinicia o ambiente e retorna o estado inicial."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent_pos = self.start_pos
        return self.state_to_idx[self.agent_pos]

    def step(self, action):
        """
        Executa uma acao no ambiente.

        Parametros
        ----------
        action : int
            Acao a executar (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)

        Retorna
        -------
        next_state : int
            Indice do proximo estado.
        reward : float
            Recompensa recebida.
        done : bool
            Se o episodio terminou.
        """
        if self.agent_pos is None:
            raise RuntimeError("Ambiente nao inicializado. Chame reset() primeiro.")

        r, c = self.agent_pos

        if self._is_terminal((r, c)):
            return self.state_to_idx[(r, c)], 0.0, True

        # Determinar acao efetiva (com estocasticidade)
        if self.slip_prob > 0 and self.rng.random() < self.slip_prob:
            # Escorrega para uma direcao perpendicular
            if action == UP or action == DOWN:
                effective_action = self.rng.choice([LEFT, RIGHT])
            else:
                effective_action = self.rng.choice([UP, DOWN])
        else:
            effective_action = action

        next_pos = self._get_next_pos(r, c, effective_action)
        reward = self._get_reward(next_pos)
        done = self._is_terminal(next_pos)

        self.agent_pos = next_pos
        return self.state_to_idx[next_pos], reward, done

    def get_state_coords(self, state_idx):
        """Retorna as coordenadas (row, col) de um estado."""
        return self.states[state_idx]

    def render(self):
        """Imprime o grid no terminal com a posicao do agente."""
        symbols = {EMPTY: ".", WALL: "W", TRAP: "T", GOAL: "G", START: "S"}
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if self.agent_pos == (r, c):
                    row_str += " A "
                else:
                    row_str += f" {symbols[self.grid[r, c]]} "
            print(row_str)
        print()


def create_frozen_lake_4x4(seed=42):
    """
    Frozen Lake 4x4 (benchmark OpenAI Gym).

    S F F F
    F H F H
    F F F H
    H F F G

    S=start, F=frozen(empty), H=hole(trap), G=goal
    slip_prob=0.333 (padrao do Gym: acao pretendida tem 1/3, cada perpendicular 1/3)
    """
    layout = [
        [START, EMPTY, EMPTY, EMPTY],
        [EMPTY, TRAP,  EMPTY, TRAP ],
        [EMPTY, EMPTY, EMPTY, TRAP ],
        [TRAP,  EMPTY, EMPTY, GOAL ],
    ]
    return GridworldEnv(grid_layout=layout, slip_prob=1/3, seed=seed)


def create_cliff_walking(seed=42):
    """
    Cliff Walking 4x12 (Sutton & Barto, Example 6.6).

    . . . . . . . . . . . .
    . . . . . . . . . . . .
    . . . . . . . . . . . .
    S T T T T T T T T T T G

    S=start (linha 3, col 0), G=goal (linha 3, col 11)
    T=cliff/trap (linha 3, cols 1-10)
    Ambiente determinístico (slip_prob=0): destaca diferenca entre
    Q-Learning (rota arriscada beira do precipício) vs SARSA/Expected SARSA
    (rota segura pela linha de cima).
    """
    H, W = 4, 12
    layout = [[EMPTY] * W for _ in range(H)]

    # Start e goal
    layout[3][0]  = START
    layout[3][11] = GOAL

    # Precipício
    for c in range(1, 11):
        layout[3][c] = TRAP

    return GridworldEnv(
        grid_layout=layout,
        slip_prob=0.0,
        reward_goal=0.0,
        reward_trap=-100.0,
        reward_step=-1.0,
        seed=seed,
    )


def create_large_gridworld(seed=42):
    """
    Cria um Gridworld 8x8 mais complexo para testes avancados.

    Layout:
    S . . . . . . .
    . W W . . W . .
    . . . . W . . .
    . W . T . . T .
    . . . . . W . .
    . T . W . . . T
    . . . . . T . .
    . . W . . . . G
    """
    layout = [
        [START, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, WALL, WALL, EMPTY, EMPTY, WALL, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, WALL, EMPTY, EMPTY, EMPTY],
        [EMPTY, WALL, EMPTY, TRAP, EMPTY, EMPTY, TRAP, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL, EMPTY, EMPTY],
        [EMPTY, TRAP, EMPTY, WALL, EMPTY, EMPTY, EMPTY, TRAP],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, TRAP, EMPTY, EMPTY],
        [EMPTY, EMPTY, WALL, EMPTY, EMPTY, EMPTY, EMPTY, GOAL],
    ]
    return GridworldEnv(grid_layout=layout, seed=seed)
