"""
Modulo de experimentacao e instrumentacao.

Responsavel por:
- Treinar agentes Q-Learning e Double Q-Learning
- Registrar metricas episodicas (recompensa, duracao, taxa de sucesso, TD error)
- Executar multiplas seeds para robustez estatistica
- Avaliar politicas aprendidas
"""

import numpy as np
from itertools import product
from gridworld_env import GridworldEnv, GOAL
from q_learning import QLearningAgent, DoubleQLearningAgent, ExpectedSARSAAgent
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
        "exploration_values": [],
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
        metrics["epsilons"].append(getattr(agent, "epsilon", 0.0))
        metrics["exploration_values"].append(agent.get_exploration_value())

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
    exploration_strategy="epsilon_greedy",
    temperature=1.0,
    temperature_min=0.05,
    temperature_decay="exponential",
    temperature_decay_rate=0.995,
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
        'q_learning', 'double_q_learning' ou 'expected_sarsa'.
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

        if agent_type == "q_learning":
            AgentClass = QLearningAgent
        elif agent_type == "double_q_learning":
            AgentClass = DoubleQLearningAgent
        elif agent_type == "expected_sarsa":
            AgentClass = ExpectedSARSAAgent
        else:
            raise ValueError(
                "agent_type invalido. Use 'q_learning', 'double_q_learning' ou 'expected_sarsa'."
            )

        agent = AgentClass(
            num_states=env.num_states,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            epsilon_decay_rate=epsilon_decay_rate,
            exploration_strategy=exploration_strategy,
            temperature=temperature,
            temperature_min=temperature_min,
            temperature_decay=temperature_decay,
            temperature_decay_rate=temperature_decay_rate,
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
    exploration_matrix = np.array([m["exploration_values"] for m in all_metrics])

    return {
        "rewards_mean": np.mean(rewards_matrix, axis=0),
        "rewards_std": np.std(rewards_matrix, axis=0),
        "steps_mean": np.mean(steps_matrix, axis=0),
        "steps_std": np.std(steps_matrix, axis=0),
        "successes_mean": np.mean(successes_matrix, axis=0),
        "td_errors_mean": np.mean(td_errors_matrix, axis=0),
        "td_errors_std": np.std(td_errors_matrix, axis=0),
        "exploration_mean": np.mean(exploration_matrix, axis=0),
        "exploration_std": np.std(exploration_matrix, axis=0),
        "num_episodes": num_episodes,
    }


def grid_search_hyperparameters(
    grid_layout,
    slip_prob,
    agent_type="q_learning",
    search_space=None,
    num_episodes=1500,
    max_steps=200,
    seeds=None,
):
    """
    Busca em grade (grid search) para hiperparametros do agente.

    A metrica de ordenacao prioriza sucesso e recompensa,
    com pequena penalizacao para trajetorias longas.

    Retorna
    -------
    best_result : dict
        Melhor configuracao encontrada.
    all_results : list of dict
        Todas as configuracoes testadas, ordenadas por score decrescente.
    """
    if seeds is None:
        seeds = [42, 123, 456]

    if search_space is None:
        search_space = {
            "alpha": [0.05, 0.1, 0.3],
            "gamma": [0.9, 0.95, 0.99],
            "epsilon_min": [0.01, 0.05],
            "epsilon_decay": ["exponential", "linear"],
            "epsilon_decay_rate": [0.995, 0.99],
        }

    keys = [
        "alpha",
        "gamma",
        "epsilon_min",
        "epsilon_decay",
        "epsilon_decay_rate",
    ]
    values = [search_space[k] for k in keys]

    all_results = []
    for combo in product(*values):
        params = dict(zip(keys, combo))
        _, eval_results = run_experiment_multiple_seeds(
            grid_layout=grid_layout,
            slip_prob=slip_prob,
            agent_type=agent_type,
            alpha=params["alpha"],
            gamma=params["gamma"],
            epsilon=1.0,
            epsilon_min=params["epsilon_min"],
            epsilon_decay=params["epsilon_decay"],
            epsilon_decay_rate=params["epsilon_decay_rate"],
            num_episodes=num_episodes,
            max_steps=max_steps,
            seeds=seeds,
            verbose=False,
        )

        mean_reward = np.mean([e["mean_reward"] for e in eval_results])
        success_rate = np.mean([e["success_rate"] for e in eval_results])
        mean_steps = np.mean([e["mean_steps"] for e in eval_results])

        score = mean_reward + 0.75 * success_rate - 0.01 * mean_steps

        all_results.append(
            {
                "params": params,
                "score": score,
                "mean_reward": mean_reward,
                "success_rate": success_rate,
                "mean_steps": mean_steps,
            }
        )

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[0], all_results
