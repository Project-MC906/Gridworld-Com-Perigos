"""
Replay Buffer para Experience Replay do DQN.

Armazena transicoes (s, a, r, s', done) e permite amostragem
aleatoria de mini-batches para quebrar correlacao temporal.
"""

import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Buffer circular de experiencias para Experience Replay.

    Parametros
    ----------
    capacity : int
        Capacidade maxima do buffer. Transicoes antigas sao descartadas.
    seed : int or None
    """

    def __init__(self, capacity=10000, seed=42):
        self.buffer = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def push(self, state, action, reward, next_state, done):
        """Adiciona uma transicao ao buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Amostra aleatoria de batch_size transicoes.

        Retorna
        -------
        states, actions, rewards, next_states, dones : np.ndarray
        """
        indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states      = np.array([t[0] for t in batch], dtype=float)
        actions     = np.array([t[1] for t in batch], dtype=int)
        rewards     = np.array([t[2] for t in batch], dtype=float)
        next_states = np.array([t[3] for t in batch], dtype=float)
        dones       = np.array([t[4] for t in batch], dtype=float)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    @property
    def is_ready(self):
        """True quando o buffer tem ao menos batch_size amostras uteis."""
        return len(self.buffer) >= 64
