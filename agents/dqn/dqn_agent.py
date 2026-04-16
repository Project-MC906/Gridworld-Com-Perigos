"""
Agente Deep Q-Network (DQN) implementado do zero com NumPy.

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
from envs.gridworld_env import NUM_ACTIONS


class DQNAgent:
    """
    Agente DQN tabular com rede neural NumPy.

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
        self._step = 0

        # Rede principal (treinada a cada passo)
        self.online_net = NeuralNetwork(
            input_size=num_states,
            hidden_sizes=hidden_sizes,
            output_size=NUM_ACTIONS,
            seed=seed,
        )

        # Target network (copia congelada, sincronizada periodicamente)
        self.target_net = NeuralNetwork(
            input_size=num_states,
            hidden_sizes=hidden_sizes,
            output_size=NUM_ACTIONS,
            seed=seed,
        )
        self.target_net.set_weights(self.online_net.get_weights())

        self.lr = lr
        self.buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)

    def _encode(self, state):
        """Codifica estado como vetor one-hot."""
        x = np.zeros(self.num_states)
        x[state] = 1.0
        return x

    def select_action(self, state):
        """Epsilon-greedy sobre a rede online."""
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, NUM_ACTIONS)
        q_values = self.online_net.forward(self._encode(state))
        return int(np.argmax(q_values))

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

        # Computar alvos com target network
        targets_all = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            q_next = self.target_net.forward(next_states[i])
            if dones[i]:
                targets_all[i] = rewards[i]
            else:
                targets_all[i] = rewards[i] + self.gamma * np.max(q_next)

        # Acumular gradientes sobre o batch
        total_loss = 0.0
        # Zerar gradientes acumulados
        for layer in self.online_net.layers:
            if hasattr(layer, 'dW'):
                layer.dW[:] = 0.0
                layer.db[:] = 0.0

        # Gradiente medio do batch
        for i in range(self.batch_size):
            q_pred_all = self.online_net.forward(states[i])

            # So otimiza Q(s, a_tomada); demais acoes nao contribuem para o gradiente
            target_vec = q_pred_all.copy()
            target_vec[actions[i]] = targets_all[i]

            loss, grad_out = self.online_net.mse_loss_and_grad(q_pred_all, target_vec)
            total_loss += loss
            self.online_net.backward(grad_out)

            # Acumular gradientes (media sera aplicada no Adam)
            for layer in self.online_net.layers:
                if hasattr(layer, 'dW'):
                    layer.dW /= self.batch_size
                    layer.db /= self.batch_size

        self.online_net.update_adam(lr=self.lr)
        return total_loss / self.batch_size

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
        policy = np.zeros(self.num_states, dtype=int)
        for s in range(self.num_states):
            q = self.online_net.forward(self._encode(s))
            policy[s] = int(np.argmax(q))
        return policy

    def get_value_function(self):
        """Retorna V(s) = max_a Q(s,a) para cada estado."""
        V = np.zeros(self.num_states)
        for s in range(self.num_states):
            q = self.online_net.forward(self._encode(s))
            V[s] = float(np.max(q))
        return V

    def get_q_table(self):
        """Retorna tabela Q(s,a) completa para comparacao com agentes tabulares."""
        Q = np.zeros((self.num_states, NUM_ACTIONS))
        for s in range(self.num_states):
            Q[s] = self.online_net.forward(self._encode(s))
        return Q
