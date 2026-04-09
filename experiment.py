"""
Modulo de experimentacao e instrumentacao.

Responsavel por:
- Treinar agentes Q-Learning e Double Q-Learning
- Registrar metricas episodicas (recompensa, duracao, taxa de sucesso, TD error)
- Executar multiplas seeds para robustez estatistica
- Avaliar politicas aprendidas
"""

import numpy as np
from gridworld_env import GridworldEnv, GOAL
from q_learning import QLearningAgent, DoubleQLearningAgent
from value_iteration import ValueIterationAgent


def train_agent(
    env,
    agent,
    num_episodes=5000,
    max_steps=200,
    verbose=False,
):
    """
    Treina um agente (Q-Learning ou Double Q-Learning) no ambiente.

    Parametros
    ----------
    env : GridworldEnv
        Ambiente de treinamento.
    agent : QLearningAgent ou DoubleQLearningAgent
        Agente a ser treinado.
    num_episodes : int
        Numero de episodios de treinamento.
    max_steps : int
        Horizonte maximo por episodio.
    verbose : bool
        Se True, imprime progresso a cada 500 episodios.

    Retorna
    -------
    metrics : dict
        Dicionario com metricas registradas:
        - 'rewards': recompensa cumulativa por episodio
        - 'steps': numero de passos por episodio
        - 'successes': 1 se atingiu o objetivo, 0 caso contrario
        - 'td_errors': media do |TD error| por episodio
        - 'epsilons': valor de epsilon ao final de cada episodio
    """
    metrics = {
        "rewards": [],
        "steps": [],
        "successes": [],
        "td_errors": [],
        "epsilons": [],
    }

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        td_errors_ep = []
        done = False

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            td_error = agent.update(state, action, reward, next_state, done)

            td_errors_ep.append(abs(td_error))
            total_reward += reward
            state = next_state

            if done:
                break

        # Decaimento de epsilon apos o episodio
        agent.decay_epsilon(episode=episode, total_episodes=num_episodes)

        # Verificar se atingiu o objetivo
        agent_pos = env.agent_pos
        success = 1 if (agent_pos is not None and env.grid[agent_pos[0], agent_pos[1]] == GOAL) else 0

        # Registrar metricas
        metrics["rewards"].append(total_reward)
        metrics["steps"].append(step + 1)
        metrics["successes"].append(success)
        metrics["td_errors"].append(np.mean(td_errors_ep) if td_errors_ep else 0.0)
        metrics["epsilons"].append(agent.epsilon)

        if verbose and (episode + 1) % 500 == 0:
            recent_rewards = metrics["rewards"][-100:]
            recent_success = metrics["successes"][-100:]
            print(
                f"  Ep {episode+1}/{num_episodes} | "
                f"Reward(100): {np.mean(recent_rewards):.3f} | "
                f"Success(100): {np.mean(recent_success):.2%} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    return metrics


def evaluate_policy(env, agent, num_episodes=100, max_steps=200):
    """
    Avalia a politica aprendida de forma gulosa (sem exploracao).

    Retorna
    -------
    results : dict
        - 'mean_reward': recompensa media
        - 'mean_steps': passos medios
        - 'success_rate': taxa de sucesso
    """
    rewards = []
    steps = []
    successes = []

    policy = agent.get_policy()

    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        for step_count in range(max_steps):
            action = policy[state]
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break

        agent_pos = env.agent_pos
        success = 1 if (agent_pos is not None and env.grid[agent_pos[0], agent_pos[1]] == GOAL) else 0

        rewards.append(total_reward)
        steps.append(step_count + 1)
        successes.append(success)

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_steps": np.mean(steps),
        "success_rate": np.mean(successes),
    }


def run_experiment_multiple_seeds(
    grid_layout=None,
    slip_prob=0.2,
    reward_goal=1.0,
    reward_trap=-1.0,
    reward_step=-0.04,
    agent_type="q_learning",
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay="exponential",
    epsilon_decay_rate=0.995,
    optimistic_init=0.0,
    num_episodes=5000,
    max_steps=200,
    seeds=None,
    verbose=False,
):
    """
    Executa o experimento com multiplas seeds e agrega resultados.

    Parametros
    ----------
    agent_type : str
        'q_learning' ou 'double_q_learning'.
    seeds : list of int
        Lista de seeds para replicacao estatistica.

    Retorna
    -------
    all_metrics : list of dict
        Lista de metricas por seed.
    eval_results : list of dict
        Lista de resultados de avaliacao por seed.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]

    all_metrics = []
    eval_results = []

    for seed in seeds:
        env = GridworldEnv(
            grid_layout=grid_layout,
            slip_prob=slip_prob,
            reward_goal=reward_goal,
            reward_trap=reward_trap,
            reward_step=reward_step,
            seed=seed,
        )

        AgentClass = DoubleQLearningAgent if agent_type == "double_q_learning" else QLearningAgent

        agent = AgentClass(
            num_states=env.num_states,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            epsilon_decay_rate=epsilon_decay_rate,
            seed=seed,
            optimistic_init=optimistic_init,
        )

        if verbose:
            print(f"\n--- Seed {seed} ({agent_type}) ---")

        metrics = train_agent(
            env, agent,
            num_episodes=num_episodes,
            max_steps=max_steps,
            verbose=verbose,
        )
        all_metrics.append(metrics)

        # Avaliar politica final
        eval_env = GridworldEnv(
            grid_layout=grid_layout,
            slip_prob=slip_prob,
            reward_goal=reward_goal,
            reward_trap=reward_trap,
            reward_step=reward_step,
            seed=seed + 10000,
        )
        eval_result = evaluate_policy(eval_env, agent, num_episodes=100, max_steps=max_steps)
        eval_results.append(eval_result)

        if verbose:
            print(
                f"  Avaliacao: Reward={eval_result['mean_reward']:.3f} +/- {eval_result['std_reward']:.3f}, "
                f"Success={eval_result['success_rate']:.2%}, "
                f"Steps={eval_result['mean_steps']:.1f}"
            )

    return all_metrics, eval_results


def aggregate_metrics(all_metrics):
    """
    Agrega metricas de multiplas seeds.

    Retorna
    -------
    aggregated : dict
        Metricas com media e desvio padrao por episodio.
    """
    num_episodes = len(all_metrics[0]["rewards"])
    rewards_matrix = np.array([m["rewards"] for m in all_metrics])
    steps_matrix = np.array([m["steps"] for m in all_metrics])
    successes_matrix = np.array([m["successes"] for m in all_metrics])
    td_errors_matrix = np.array([m["td_errors"] for m in all_metrics])

    return {
        "rewards_mean": np.mean(rewards_matrix, axis=0),
        "rewards_std": np.std(rewards_matrix, axis=0),
        "steps_mean": np.mean(steps_matrix, axis=0),
        "steps_std": np.std(steps_matrix, axis=0),
        "successes_mean": np.mean(successes_matrix, axis=0),
        "td_errors_mean": np.mean(td_errors_matrix, axis=0),
        "td_errors_std": np.std(td_errors_matrix, axis=0),
        "num_episodes": num_episodes,
    }
