"""
Agente Deep Q-Network (DQN) implementado do zero com NumPy/CuPy.

Implementa:
- Rede neural para aproximar Q(s,a)
- Target network (copia congelada, sincronizada periodicamente)
- Experience Replay com mini-batch
- Politica epsilon-greedy com decaimento
- Codificacao one-hot dos estados

Referencias:
    Mnih et al. (2015) - Human-level control through deep reinforcement learning
"""

import numpy as np
from agents.dqn.neural_net import NeuralNetwork
from agents.dqn.replay_buffer import ReplayBuffer
from agents.dqn.device import get_array_module, to_numpy
from envs.gridworld_env import NUM_ACTIONS


class DQNAgent:
    """
    Agente DQN tabular com rede neural NumPy/CuPy.

    Parametros
    ----------
    num_states : int
        Numero de estados (tamanho do vetor one-hot de entrada).
    hidden_sizes : list of int
        Tamanhos das camadas ocultas da rede.
    lr : float
        Taxa de aprendizado do otimizador Adam.
    gamma : float
        Fator de desconto.
    epsilon : float
        Epsilon inicial para exploracao.
    epsilon_min : float
        Piso de epsilon.
    epsilon_decay : str
        'exponential' ou 'linear'.
    epsilon_decay_rate : float
        Taxa de decaimento exponencial.
    batch_size : int
        Tamanho do mini-batch para treinamento.
    buffer_capacity : int
        Capacidade do replay buffer.
    target_update_freq : int
        A cada quantos passos sincronizar a target network.
    device : str
        "auto", "cpu" ou "cuda".
    seed : int or None
    """

    def __init__(
        self,
        num_states,
        hidden_sizes=None,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay="exponential",
        epsilon_decay_rate=0.995,
        batch_size=64,
        buffer_capacity=10000,
        target_update_freq=100,
        device="auto",
        seed=42,
        # Parametros ignorados (compatibilidade com interface tabular)
        alpha=None,
        optimistic_init=None,
        exploration_strategy=None,
        temperature=None,
        temperature_min=None,
        temperature_decay=None,
        temperature_decay_rate=None,
        epsilon_min_=None,
    ):
        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        self.num_states = num_states
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_rate = epsilon_decay_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.rng = np.random.default_rng(seed)
        self.xp, self.device = get_array_module(device)
        self._step = 0

        # Cache one-hot em CPU para evitar alocacao repetida.
        self._identity_cache = np.eye(self.num_states, dtype=np.float32)

        # Rede principal (treinada a cada passo)
        self.online_net = NeuralNetwork(
            input_size=num_states,
            hidden_sizes=hidden_sizes,
            output_size=NUM_ACTIONS,
            xp=self.xp,
            seed=seed,
        )

        # Target network (copia congelada, sincronizada periodicamente)
        self.target_net = NeuralNetwork(
            input_size=num_states,
            hidden_sizes=hidden_sizes,
            output_size=NUM_ACTIONS,
            xp=self.xp,
            seed=seed,
        )
        self.target_net.set_weights(self.online_net.get_weights())

        self.lr = lr
        self.buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)

    def _encode(self, state):
        """Codifica estado como vetor one-hot em memoria host (CPU)."""
        return self._identity_cache[state]

    def _to_device(self, array):
        """Move dados para o backend numerico ativo (CPU/CUDA)."""
        return self.xp.asarray(array, dtype=self.xp.float32)

    def select_action(self, state):
        """Epsilon-greedy sobre a rede online."""
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, NUM_ACTIONS)

        q_values = self.online_net.forward(self._to_device(self._encode(state)))
        return int(to_numpy(self.xp.argmax(q_values)))

    def update(self, state, action, reward, next_state, done):
        """
        Armazena transicao e executa um passo de treinamento se o buffer
        tiver amostras suficientes.

        Retorna
        -------
        loss : float
            MSE loss do batch (0.0 se buffer ainda nao esta pronto).
        """
        self.buffer.push(
            self._encode(state), action, reward,
            self._encode(next_state), float(done),
        )
        self._step += 1

        if len(self.buffer) < self.batch_size:
            return 0.0

        loss = self._train_step()

        # Sincronizar target network periodicamente
        if self._step % self.target_update_freq == 0:
            self.target_net.set_weights(self.online_net.get_weights())

        return loss

    def _train_step(self):
        """
        Um passo de treinamento com mini-batch do replay buffer.

        Alvo DQN:
            y = r                          se done
            y = r + gamma * max_a Q_target(s', a)  caso contrario

        Loss: MSE(Q_online(s,a), y)
        """
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_dev = self._to_device(states)
        actions_dev = self.xp.asarray(actions, dtype=self.xp.int64)
        rewards_dev = self._to_device(rewards)
        next_states_dev = self._to_device(next_states)
        dones_dev = self._to_device(dones)

        # y = r + gamma * max_a Q_target(s',a) (ou y = r quando done)
        q_next_all = self.target_net.forward(next_states_dev)
        max_q_next = self.xp.max(q_next_all, axis=1)
        targets = rewards_dev + (1.0 - dones_dev) * self.gamma * max_q_next

        # Treina apenas a acao executada em cada transicao do batch.
        q_pred_all = self.online_net.forward(states_dev)
        target_vec = q_pred_all.copy()
        batch_idx = self.xp.arange(self.batch_size, dtype=self.xp.int64)
        target_vec[batch_idx, actions_dev] = targets

        loss, grad_out = self.online_net.mse_loss_and_grad(q_pred_all, target_vec)
        self.online_net.zero_grad()
        self.online_net.backward(grad_out)
        self.online_net.update_adam(lr=self.lr)
        return float(to_numpy(loss))

    def decay_epsilon(self, episode=None, total_episodes=None):
        """Decai epsilon ao final de cada episodio."""
        if self.epsilon_decay == "exponential":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)
        elif self.epsilon_decay == "linear":
            if total_episodes:
                step = (self.epsilon_start - self.epsilon_min) / total_episodes
                self.epsilon = max(self.epsilon_min, self.epsilon - step)

    def get_exploration_value(self):
        return self.epsilon

    def get_policy(self):
        """Retorna politica gulosa: argmax_a Q(s,a) para cada estado."""
        q_table = self.get_q_table()
        return np.argmax(q_table, axis=1)

    def get_value_function(self):
        """Retorna V(s) = max_a Q(s,a) para cada estado."""
        q_table = self.get_q_table()
        return np.max(q_table, axis=1)

    def get_q_table(self):
        """Retorna tabela Q(s,a) completa para comparacao com agentes tabulares."""
        states_dev = self._to_device(self._identity_cache)
        q_values = self.online_net.forward(states_dev)
        return to_numpy(q_values)
