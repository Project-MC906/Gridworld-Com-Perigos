"""
Modulo de Value Iteration (Iteracao de Valor).

Algoritmo de programacao dinamica baseado em modelo que resolve o MDP
usando as Equacoes de Otimalidade de Bellman.
Requer acesso completo ao modelo de transicao P(s'|s,a) e R(s,a,s').
"""

import numpy as np
from gridworld_env import NUM_ACTIONS


class ValueIterationAgent:
    """
    Agente que computa a politica otima via Value Iteration.

    Parametros
    ----------
    env : GridworldEnv
        Ambiente Gridworld com modelo de transicao acessivel.
    gamma : float
        Fator de desconto (0 < gamma <= 1).
    theta : float
        Limiar de convergencia (norma infinita da variacao de V).
    """

    def __init__(self, env, gamma=0.99, theta=1e-8):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.num_states = env.num_states
        self.V = np.zeros(self.num_states)
        self.policy = np.zeros(self.num_states, dtype=int)
        self.converged = False
        self.iterations = 0
        self.delta_history = []  # historico de delta por iteracao

    def run(self, max_iterations=10000):
        """
        Executa Value Iteration ate convergencia ou limite de iteracoes.

        Retorna
        -------
        V : np.ndarray
            Funcao de valor otima.
        policy : np.ndarray
            Politica otima (acao por estado).
        iterations : int
            Numero de iteracoes ate convergencia.
        """
        for i in range(max_iterations):
            delta = 0.0
            V_new = np.copy(self.V)

            for s in range(self.num_states):
                # Pular estados terminais
                pos = self.env.states[s]
                if self.env._is_terminal(pos):
                    continue

                # Calcular Q(s,a) para todas as acoes
                q_values = np.zeros(NUM_ACTIONS)
                for a in range(NUM_ACTIONS):
                    for prob, next_s, reward, done in self.env.P[s][a]:
                        if done:
                            q_values[a] += prob * reward
                        else:
                            q_values[a] += prob * (reward + self.gamma * self.V[next_s])

                # Atualizar V(s) = max_a Q(s,a)
                best_value = np.max(q_values)
                delta = max(delta, abs(best_value - self.V[s]))
                V_new[s] = best_value

            self.V = V_new
            self.iterations = i + 1
            self.delta_history.append(delta)

            # Criterio de convergencia: norma infinita < theta
            if delta < self.theta:
                self.converged = True
                break

        # Extrair politica otima via lookahead de passo unico
        self._extract_policy()
        return self.V, self.policy, self.iterations

    def _extract_policy(self):
        """
        Extrai a politica otima a partir da funcao de valor convergida.

        pi*(s) = argmax_a sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V*(s')]
        """
        for s in range(self.num_states):
            pos = self.env.states[s]
            if self.env._is_terminal(pos):
                self.policy[s] = 0  # irrelevante para estados terminais
                continue

            q_values = np.zeros(NUM_ACTIONS)
            for a in range(NUM_ACTIONS):
                for prob, next_s, reward, done in self.env.P[s][a]:
                    if done:
                        q_values[a] += prob * reward
                    else:
                        q_values[a] += prob * (reward + self.gamma * self.V[next_s])

            self.policy[s] = np.argmax(q_values)

    def get_q_values(self):
        """
        Computa a tabela Q(s,a) completa a partir de V*.
        Util para comparacoes com Q-learning.
        """
        Q = np.zeros((self.num_states, NUM_ACTIONS))
        for s in range(self.num_states):
            pos = self.env.states[s]
            if self.env._is_terminal(pos):
                continue
            for a in range(NUM_ACTIONS):
                for prob, next_s, reward, done in self.env.P[s][a]:
                    if done:
                        Q[s, a] += prob * reward
                    else:
                        Q[s, a] += prob * (reward + self.gamma * self.V[next_s])
        return Q

    def get_policy(self):
        """
        Retorna a politica atual.

        Se a politica ainda nao foi extraida (por exemplo, antes de run()),
        extrai a partir da funcao de valor corrente.
        """
        self._extract_policy()
        return self.policy

    def get_value_function(self):
        """Retorna a funcao de valor atual V(s)."""
        return self.V
