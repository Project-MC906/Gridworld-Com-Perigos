"""
Modulo de Q-Learning e Double Q-Learning.

Algoritmos de aprendizado por diferenca temporal (TD) livres de modelo.
O agente aprende exclusivamente pela interacao com o ambiente.
"""

import numpy as np
from envs.gridworld_env import NUM_ACTIONS


class QLearningAgent:
    """
    Agente Q-Learning tabular com politica epsilon-greedy.

    Parametros
    ----------
    num_states : int
        Numero de estados no ambiente.
    alpha : float
        Taxa de aprendizado.
    gamma : float
        Fator de desconto.
    epsilon : float
        Parametro de exploracao epsilon-greedy inicial.
    epsilon_min : float
        Piso minimo para epsilon.
    epsilon_decay : str
        Estrategia de decaimento: 'none', 'linear', 'exponential'.
    epsilon_decay_rate : float
        Taxa de decaimento (para exponencial) ou numero total de episodios para
        atingir epsilon_min (para linear).
    exploration_strategy : str
        Estrategia de exploracao: 'epsilon_greedy' ou 'softmax'.
    temperature : float
        Temperatura inicial da politica softmax (Boltzmann).
    temperature_min : float
        Piso minimo para temperatura na estrategia softmax.
    temperature_decay : str
        Estrategia de decaimento da temperatura: 'none', 'linear', 'exponential'.
    temperature_decay_rate : float
        Taxa de decaimento da temperatura (para exponencial).
    seed : int or None
        Semente para reprodutibilidade.
    optimistic_init : float
        Valor de inicializacao otimista para a tabela Q. Se 0, inicializa com zeros.
    """

    def __init__(
        self,
        num_states,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay="exponential",
        epsilon_decay_rate=0.995,
        exploration_strategy="epsilon_greedy",
        temperature=1.0,
        temperature_min=0.05,
        temperature_decay="exponential",
        temperature_decay_rate=0.995,
        seed=42,
        optimistic_init=0.0,
    ):
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_rate = epsilon_decay_rate
        self.exploration_strategy = exploration_strategy
        self.temperature = temperature
        self.temperature_start = temperature
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        self.temperature_decay_rate = temperature_decay_rate
        self.rng = np.random.default_rng(seed)

        # Inicializacao da tabela Q
        if optimistic_init > 0:
            self.Q = np.full((num_states, NUM_ACTIONS), optimistic_init, dtype=float)
        else:
            self.Q = np.zeros((num_states, NUM_ACTIONS), dtype=float)

    def select_action(self, state):
        """
        Seleciona acao usando estrategia de exploracao configurada.

        - epsilon_greedy: com probabilidade epsilon, escolhe acao aleatoria.
        - softmax: amostra acao por distribuicao de Boltzmann sobre Q(s,a).
        """
        if self.exploration_strategy == "epsilon_greedy":
            if self.rng.random() < self.epsilon:
                return self.rng.integers(0, NUM_ACTIONS)

            # Desempate aleatorio entre acoes de mesmo valor
            q_values = self.Q[state]
            max_q = np.max(q_values)
            best_actions = np.where(np.isclose(q_values, max_q))[0]
            return self.rng.choice(best_actions)

        if self.exploration_strategy == "softmax":
            q_values = self.Q[state]
            probs = self._softmax_probs(q_values)
            return self.rng.choice(np.arange(NUM_ACTIONS), p=probs)

        raise ValueError(
            "exploration_strategy invalida. Use 'epsilon_greedy' ou 'softmax'."
        )

    def _softmax_probs(self, q_values):
        """Calcula probabilidades da politica de Boltzmann para um vetor Q."""
        temp = max(self.temperature, 1e-8)
        shifted = q_values - np.max(q_values)
        exps = np.exp(shifted / temp)
        return exps / np.sum(exps)

    def update(self, state, action, reward, next_state, done):
        """
        Atualiza Q(s,a) pela regra de Q-Learning:

        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        Retorna
        -------
        td_error : float
            Erro de diferenca temporal.
        """
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])

        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return td_error

    def decay_epsilon(self, episode=None, total_episodes=None):
        """
        Aplica decaimento no parametro de exploracao ao final de cada episodio.

        Parametros
        ----------
        episode : int
            Episodio atual (para decaimento linear).
        total_episodes : int
            Total de episodios (para decaimento linear).
        """
        if self.exploration_strategy == "epsilon_greedy":
            if self.epsilon_decay == "exponential":
                self.epsilon = max(
                    self.epsilon_min, self.epsilon * self.epsilon_decay_rate
                )
            elif self.epsilon_decay == "linear":
                if total_episodes is not None and total_episodes > 0:
                    decay_per_episode = (self.epsilon_start - self.epsilon_min) / total_episodes
                    self.epsilon = max(self.epsilon_min, self.epsilon - decay_per_episode)
            # 'none': epsilon fixo, nao decai
            return

        if self.exploration_strategy == "softmax":
            if self.temperature_decay == "exponential":
                self.temperature = max(
                    self.temperature_min,
                    self.temperature * self.temperature_decay_rate,
                )
            elif self.temperature_decay == "linear":
                if total_episodes is not None and total_episodes > 0:
                    decay_per_episode = (
                        self.temperature_start - self.temperature_min
                    ) / total_episodes
                    self.temperature = max(
                        self.temperature_min,
                        self.temperature - decay_per_episode,
                    )
            # 'none': temperatura fixa, nao decai
            return

    def get_exploration_value(self):
        """Retorna o parametro de exploracao atual (epsilon ou temperatura)."""
        if self.exploration_strategy == "softmax":
            return self.temperature
        return self.epsilon

    def get_policy(self):
        """Retorna a politica gulosa derivada de Q."""
        return np.argmax(self.Q, axis=1)

    def get_value_function(self):
        """Retorna V(s) = max_a Q(s,a)."""
        return np.max(self.Q, axis=1)


class ExpectedSARSAAgent:
    """
    Agente Expected SARSA tabular com politica epsilon-greedy.

    Diferente do Q-Learning, o alvo TD usa a expectativa sobre a politica
    no proximo estado, reduzindo variancia da atualizacao.
    """

    def __init__(
        self,
        num_states,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay="exponential",
        epsilon_decay_rate=0.995,
        exploration_strategy="epsilon_greedy",
        temperature=1.0,
        temperature_min=0.05,
        temperature_decay="exponential",
        temperature_decay_rate=0.995,
        seed=42,
        optimistic_init=0.0,
    ):
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_rate = epsilon_decay_rate
        self.exploration_strategy = exploration_strategy
        self.temperature = temperature
        self.temperature_start = temperature
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        self.temperature_decay_rate = temperature_decay_rate
        self.rng = np.random.default_rng(seed)

        if optimistic_init > 0:
            self.Q = np.full((num_states, NUM_ACTIONS), optimistic_init, dtype=float)
        else:
            self.Q = np.zeros((num_states, NUM_ACTIONS), dtype=float)

    def select_action(self, state):
        """Seleciona acao usando estrategia de exploracao configurada."""
        if self.exploration_strategy == "epsilon_greedy":
            if self.rng.random() < self.epsilon:
                return self.rng.integers(0, NUM_ACTIONS)

            q_values = self.Q[state]
            max_q = np.max(q_values)
            best_actions = np.where(np.isclose(q_values, max_q))[0]
            return self.rng.choice(best_actions)

        if self.exploration_strategy == "softmax":
            probs = self._softmax_probs(self.Q[state])
            return self.rng.choice(np.arange(NUM_ACTIONS), p=probs)

        raise ValueError(
            "exploration_strategy invalida. Use 'epsilon_greedy' ou 'softmax'."
        )

    def _softmax_probs(self, q_values):
        """Distribuicao de probabilidades da politica softmax."""
        temp = max(self.temperature, 1e-8)
        shifted = q_values - np.max(q_values)
        exps = np.exp(shifted / temp)
        return exps / np.sum(exps)

    def _epsilon_greedy_probs(self, state):
        """Distribuicao de probabilidade da politica epsilon-greedy em um estado."""
        probs = np.full(NUM_ACTIONS, self.epsilon / NUM_ACTIONS, dtype=float)
        q_values = self.Q[state]
        max_q = np.max(q_values)
        best_actions = np.where(np.isclose(q_values, max_q))[0]
        probs[best_actions] += (1.0 - self.epsilon) / len(best_actions)
        return probs

    def update(self, state, action, reward, next_state, done):
        """
        Atualiza Q(s,a) pela regra de Expected SARSA:

        Q(s,a) <- Q(s,a) + alpha * [r + gamma * E_{a'~pi}[Q(s',a')] - Q(s,a)]
        """
        if done:
            td_target = reward
        else:
            if self.exploration_strategy == "epsilon_greedy":
                action_probs = self._epsilon_greedy_probs(next_state)
            elif self.exploration_strategy == "softmax":
                action_probs = self._softmax_probs(self.Q[next_state])
            else:
                raise ValueError(
                    "exploration_strategy invalida. Use 'epsilon_greedy' ou 'softmax'."
                )
            expected_next_q = np.dot(action_probs, self.Q[next_state])
            td_target = reward + self.gamma * expected_next_q

        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return td_error

    def decay_epsilon(self, episode=None, total_episodes=None):
        """Aplica decaimento no parametro de exploracao no final de cada episodio."""
        if self.exploration_strategy == "epsilon_greedy":
            if self.epsilon_decay == "exponential":
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)
            elif self.epsilon_decay == "linear":
                if total_episodes is not None and total_episodes > 0:
                    decay_per_episode = (self.epsilon_start - self.epsilon_min) / total_episodes
                    self.epsilon = max(self.epsilon_min, self.epsilon - decay_per_episode)
            return

        if self.exploration_strategy == "softmax":
            if self.temperature_decay == "exponential":
                self.temperature = max(
                    self.temperature_min,
                    self.temperature * self.temperature_decay_rate,
                )
            elif self.temperature_decay == "linear":
                if total_episodes is not None and total_episodes > 0:
                    decay_per_episode = (
                        self.temperature_start - self.temperature_min
                    ) / total_episodes
                    self.temperature = max(
                        self.temperature_min,
                        self.temperature - decay_per_episode,
                    )

    def get_exploration_value(self):
        """Retorna o parametro de exploracao atual (epsilon ou temperatura)."""
        if self.exploration_strategy == "softmax":
            return self.temperature
        return self.epsilon

    def get_policy(self):
        """Retorna a politica gulosa derivada de Q."""
        return np.argmax(self.Q, axis=1)

    def get_value_function(self):
        """Retorna V(s) = max_a Q(s,a)."""
        return np.max(self.Q, axis=1)


class DoubleQLearningAgent:
    """
    Agente Double Q-Learning tabular.

    Mantem duas tabelas Q independentes (Q1, Q2) para mitigar o vies
    de superestimacao do Q-Learning classico.

    A cada passo, seleciona aleatoriamente qual tabela atualizar.
    A tabela selecionada escolhe a melhor acao, mas o valor eh
    avaliado pela outra tabela (desacoplamento selecao-avaliacao).
    """

    def __init__(
        self,
        num_states,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay="exponential",
        epsilon_decay_rate=0.995,
        exploration_strategy="epsilon_greedy",
        temperature=1.0,
        temperature_min=0.05,
        temperature_decay="exponential",
        temperature_decay_rate=0.995,
        seed=42,
        optimistic_init=0.0,
    ):
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_rate = epsilon_decay_rate
        self.exploration_strategy = exploration_strategy
        self.temperature = temperature
        self.temperature_start = temperature
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        self.temperature_decay_rate = temperature_decay_rate
        self.rng = np.random.default_rng(seed)

        # Duas tabelas Q independentes
        if optimistic_init > 0:
            self.Q1 = np.full((num_states, NUM_ACTIONS), optimistic_init, dtype=float)
            self.Q2 = np.full((num_states, NUM_ACTIONS), optimistic_init, dtype=float)
        else:
            self.Q1 = np.zeros((num_states, NUM_ACTIONS), dtype=float)
            self.Q2 = np.zeros((num_states, NUM_ACTIONS), dtype=float)

    def select_action(self, state):
        """
        Seleciona acao usando estrategia de exploracao sobre Q1 + Q2.
        """
        q_combined = self.Q1[state] + self.Q2[state]

        if self.exploration_strategy == "epsilon_greedy":
            if self.rng.random() < self.epsilon:
                return self.rng.integers(0, NUM_ACTIONS)

            max_q = np.max(q_combined)
            best_actions = np.where(np.isclose(q_combined, max_q))[0]
            return self.rng.choice(best_actions)

        if self.exploration_strategy == "softmax":
            probs = self._softmax_probs(q_combined)
            return self.rng.choice(np.arange(NUM_ACTIONS), p=probs)

        raise ValueError(
            "exploration_strategy invalida. Use 'epsilon_greedy' ou 'softmax'."
        )

    def _softmax_probs(self, q_values):
        """Distribuicao de probabilidades da politica softmax."""
        temp = max(self.temperature, 1e-8)
        shifted = q_values - np.max(q_values)
        exps = np.exp(shifted / temp)
        return exps / np.sum(exps)

    def update(self, state, action, reward, next_state, done):
        """
        Atualiza uma das tabelas Q (selecionada aleatoriamente).

        Se Q1 eh selecionada:
            a* = argmax_a Q1(s', a)
            Q1(s,a) <- Q1(s,a) + alpha * [r + gamma * Q2(s', a*) - Q1(s,a)]

        Analogamente para Q2.

        Retorna
        -------
        td_error : float
            Erro de diferenca temporal.
        """
        if self.rng.random() < 0.5:
            # Atualizar Q1
            if done:
                td_target = reward
            else:
                best_action = np.argmax(self.Q1[next_state])
                td_target = reward + self.gamma * self.Q2[next_state, best_action]

            td_error = td_target - self.Q1[state, action]
            self.Q1[state, action] += self.alpha * td_error
        else:
            # Atualizar Q2
            if done:
                td_target = reward
            else:
                best_action = np.argmax(self.Q2[next_state])
                td_target = reward + self.gamma * self.Q1[next_state, best_action]

            td_error = td_target - self.Q2[state, action]
            self.Q2[state, action] += self.alpha * td_error

        return td_error

    def decay_epsilon(self, episode=None, total_episodes=None):
        """Aplica decaimento no parametro de exploracao no final de cada episodio."""
        if self.exploration_strategy == "epsilon_greedy":
            if self.epsilon_decay == "exponential":
                self.epsilon = max(
                    self.epsilon_min, self.epsilon * self.epsilon_decay_rate
                )
            elif self.epsilon_decay == "linear":
                if total_episodes is not None and total_episodes > 0:
                    decay_per_episode = (self.epsilon_start - self.epsilon_min) / total_episodes
                    self.epsilon = max(self.epsilon_min, self.epsilon - decay_per_episode)
            return

        if self.exploration_strategy == "softmax":
            if self.temperature_decay == "exponential":
                self.temperature = max(
                    self.temperature_min,
                    self.temperature * self.temperature_decay_rate,
                )
            elif self.temperature_decay == "linear":
                if total_episodes is not None and total_episodes > 0:
                    decay_per_episode = (
                        self.temperature_start - self.temperature_min
                    ) / total_episodes
                    self.temperature = max(
                        self.temperature_min,
                        self.temperature - decay_per_episode,
                    )

    def get_exploration_value(self):
        """Retorna o parametro de exploracao atual (epsilon ou temperatura)."""
        if self.exploration_strategy == "softmax":
            return self.temperature
        return self.epsilon

    def get_policy(self):
        """Retorna a politica gulosa derivada de (Q1 + Q2) / 2."""
        Q_avg = (self.Q1 + self.Q2) / 2.0
        return np.argmax(Q_avg, axis=1)

    def get_value_function(self):
        """Retorna V(s) = max_a (Q1 + Q2)(s,a) / 2."""
        Q_avg = (self.Q1 + self.Q2) / 2.0
        return np.max(Q_avg, axis=1)

    def get_q_table(self):
        """Retorna a tabela Q combinada (Q1 + Q2) / 2."""
        return (self.Q1 + self.Q2) / 2.0
