"""
Script principal para execucao dos experimentos de RL no Gridworld com Perigos.

Executa:
1. Value Iteration (planejamento off-line com modelo completo)
2. Q-Learning (aprendizado empírico livre de modelo)
3. Double Q-Learning (mitigacao do vies de superestimacao)
4. Expected SARSA (controle on-policy em modelo livre)
5. Comparacoes de hiperparametros (incluindo grid search)
6. Geracao de todas as visualizacoes

Uso:
    python main.py
"""

import numpy as np
import os

from gridworld_env import GridworldEnv, create_large_gridworld
from value_iteration import ValueIterationAgent
from q_learning import QLearningAgent, DoubleQLearningAgent
from experiment import (
    train_agent,
    evaluate_policy,
    run_experiment_multiple_seeds,
    aggregate_metrics,
    grid_search_hyperparameters,
)
from visualization import (
    plot_value_heatmap,
    plot_policy_arrows,
    plot_value_and_policy,
    plot_learning_curves,
    plot_convergence_vi,
    plot_trajectory,
    plot_comparison_bar,
)

# ============================================================
# Configuracoes globais
# ============================================================
OUTPUT_DIR = "resultados"
SEEDS = [42, 123, 456, 789, 1024]
NUM_EPISODES = 5000
MAX_STEPS = 200

# Gridworld 4x4 padrao
GRID_4x4 = None  # usa layout padrao do GridworldEnv

# Gridworld 8x8
GRID_8x8 = [
    [4, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 2, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 2, 0, 1, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 3],
]


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PARTE 1: Value Iteration
# ============================================================
def run_value_iteration_experiments():
    print("=" * 70)
    print("PARTE 1: VALUE ITERATION (Planejamento com modelo completo)")
    print("=" * 70)

    for grid_name, grid_layout, slip in [
        ("4x4_det", GRID_4x4, 0.0),
        ("4x4_stoch", GRID_4x4, 0.2),
        ("8x8_stoch", GRID_8x8, 0.2),
    ]:
        print(f"\n--- Value Iteration: {grid_name} (slip={slip}) ---")
        env = GridworldEnv(grid_layout=grid_layout, slip_prob=slip, seed=42)

        # Testar diferentes gammas
        for gamma in [0.5, 0.9, 0.99]:
            agent = ValueIterationAgent(env, gamma=gamma, theta=1e-8)
            V, policy, iterations = agent.run()

            print(
                f"  gamma={gamma}: Convergiu em {iterations} iteracoes | "
                f"V(start)={V[env.state_to_idx[env.start_pos]]:.4f}"
            )

            # Visualizacoes
            plot_value_and_policy(
                env, V, policy,
                suptitle=f"Value Iteration - {grid_name} (gamma={gamma}, slip={slip})",
                filename=f"{OUTPUT_DIR}/vi_{grid_name}_gamma{gamma}.png",
            )

            plot_convergence_vi(
                agent.delta_history,
                title=f"Convergencia VI - {grid_name} (gamma={gamma})",
                filename=f"{OUTPUT_DIR}/vi_convergence_{grid_name}_gamma{gamma}.png",
            )

            # Trajetoria
            plot_trajectory(
                env, policy,
                title=f"Trajetoria VI - {grid_name} (gamma={gamma})",
                filename=f"{OUTPUT_DIR}/vi_trajectory_{grid_name}_gamma{gamma}.png",
            )


# ============================================================
# PARTE 2: Q-Learning vs Double Q-Learning
# ============================================================
def run_qlearning_experiments():
    print("\n" + "=" * 70)
    print("PARTE 2: Q-LEARNING vs DOUBLE Q-LEARNING vs EXPECTED SARSA")
    print("=" * 70)

    for grid_name, grid_layout, slip in [
        ("4x4_stoch", GRID_4x4, 0.2),
        ("8x8_stoch", GRID_8x8, 0.2),
    ]:
        print(f"\n--- Comparacao Q vs Double-Q vs Expected SARSA: {grid_name} (slip={slip}) ---")

        all_aggregated = {}
        all_eval = {}

        for agent_type, label in [
            ("q_learning", "Q-Learning"),
            ("double_q_learning", "Double Q-Learning"),
            ("expected_sarsa", "Expected SARSA"),
        ]:
            metrics_list, eval_list = run_experiment_multiple_seeds(
                grid_layout=grid_layout,
                slip_prob=slip,
                agent_type=agent_type,
                alpha=0.1,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay="exponential",
                epsilon_decay_rate=0.995,
                num_episodes=NUM_EPISODES,
                max_steps=MAX_STEPS,
                seeds=SEEDS,
                verbose=True,
            )

            agg = aggregate_metrics(metrics_list)
            all_aggregated[label] = agg

            # Media das avaliacoes
            avg_eval = {
                "mean_reward": np.mean([e["mean_reward"] for e in eval_list]),
                "success_rate": np.mean([e["success_rate"] for e in eval_list]),
                "mean_steps": np.mean([e["mean_steps"] for e in eval_list]),
            }
            all_eval[label] = avg_eval
            print(
                f"\n  {label} - Avaliacao media: "
                f"Reward={avg_eval['mean_reward']:.3f}, "
                f"Success={avg_eval['success_rate']:.2%}, "
                f"Steps={avg_eval['mean_steps']:.1f}"
            )

        # Curvas de aprendizado comparativas
        plot_learning_curves(
            all_aggregated,
            title_prefix=f"Q-Learning vs Double Q-Learning vs Expected SARSA - {grid_name}",
            filename=f"{OUTPUT_DIR}/comparison_q_vs_dq_vs_es_{grid_name}.png",
        )

        # Barras comparativas
        plot_comparison_bar(
            all_eval,
            filename=f"{OUTPUT_DIR}/eval_bar_{grid_name}.png",
        )

        # Visualizar politica final de referencia (Q-Learning)
        env = GridworldEnv(grid_layout=grid_layout, slip_prob=slip, seed=42)
        agent = QLearningAgent(
            num_states=env.num_states, alpha=0.1, gamma=0.99,
            epsilon=1.0, epsilon_min=0.01, epsilon_decay="exponential",
            epsilon_decay_rate=0.995, seed=42,
        )
        train_agent(env, agent, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)
        V = agent.get_value_function()
        policy = agent.get_policy()

        plot_value_and_policy(
            env, V, policy,
            suptitle=f"Q-Learning Final - {grid_name}",
            filename=f"{OUTPUT_DIR}/qlearning_final_{grid_name}.png",
        )
        plot_trajectory(
            env, policy,
            title=f"Trajetoria Q-Learning - {grid_name}",
            filename=f"{OUTPUT_DIR}/qlearning_trajectory_{grid_name}.png",
        )


# ============================================================
# PARTE 3: Estudo de Hiperparametros
# ============================================================
def run_hyperparameter_study():
    print("\n" + "=" * 70)
    print("PARTE 3: ESTUDO DE HIPERPARAMETROS")
    print("=" * 70)

    grid_layout = GRID_4x4
    slip = 0.2

    # --- 3a. Efeito do Gamma ---
    print("\n--- 3a. Efeito do Fator de Desconto (gamma) ---")
    gamma_aggregated = {}
    for gamma in [0.5, 0.9, 0.95, 0.99]:
        metrics_list, _ = run_experiment_multiple_seeds(
            grid_layout=grid_layout, slip_prob=slip,
            agent_type="q_learning", alpha=0.1, gamma=gamma,
            epsilon=1.0, epsilon_min=0.01,
            epsilon_decay="exponential", epsilon_decay_rate=0.995,
            num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, seeds=SEEDS,
        )
        gamma_aggregated[f"gamma={gamma}"] = aggregate_metrics(metrics_list)

    plot_learning_curves(
        gamma_aggregated,
        title_prefix="Efeito do Fator de Desconto (gamma)",
        filename=f"{OUTPUT_DIR}/hp_gamma_comparison.png",
    )

    # --- 3b. Efeito do Alpha ---
    print("\n--- 3b. Efeito da Taxa de Aprendizado (alpha) ---")
    alpha_aggregated = {}
    for alpha in [0.01, 0.05, 0.1, 0.3, 0.5]:
        metrics_list, _ = run_experiment_multiple_seeds(
            grid_layout=grid_layout, slip_prob=slip,
            agent_type="q_learning", alpha=alpha, gamma=0.99,
            epsilon=1.0, epsilon_min=0.01,
            epsilon_decay="exponential", epsilon_decay_rate=0.995,
            num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, seeds=SEEDS,
        )
        alpha_aggregated[f"alpha={alpha}"] = aggregate_metrics(metrics_list)

    plot_learning_curves(
        alpha_aggregated,
        title_prefix="Efeito da Taxa de Aprendizado (alpha)",
        filename=f"{OUTPUT_DIR}/hp_alpha_comparison.png",
    )

    # --- 3c. Estrategias de Decaimento de Epsilon ---
    print("\n--- 3c. Estrategias de Decaimento de Epsilon ---")
    eps_aggregated = {}

    # Epsilon fixo
    metrics_list, _ = run_experiment_multiple_seeds(
        grid_layout=grid_layout, slip_prob=slip,
        agent_type="q_learning", alpha=0.1, gamma=0.99,
        epsilon=0.1, epsilon_min=0.1,
        epsilon_decay="none",
        num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, seeds=SEEDS,
    )
    eps_aggregated["Fixo (eps=0.1)"] = aggregate_metrics(metrics_list)

    # Decaimento linear
    metrics_list, _ = run_experiment_multiple_seeds(
        grid_layout=grid_layout, slip_prob=slip,
        agent_type="q_learning", alpha=0.1, gamma=0.99,
        epsilon=1.0, epsilon_min=0.01,
        epsilon_decay="linear",
        num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, seeds=SEEDS,
    )
    eps_aggregated["Linear (1.0 -> 0.01)"] = aggregate_metrics(metrics_list)

    # Decaimento exponencial
    metrics_list, _ = run_experiment_multiple_seeds(
        grid_layout=grid_layout, slip_prob=slip,
        agent_type="q_learning", alpha=0.1, gamma=0.99,
        epsilon=1.0, epsilon_min=0.01,
        epsilon_decay="exponential", epsilon_decay_rate=0.995,
        num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, seeds=SEEDS,
    )
    eps_aggregated["Exponencial (rate=0.995)"] = aggregate_metrics(metrics_list)

    # Decaimento exponencial mais agressivo
    metrics_list, _ = run_experiment_multiple_seeds(
        grid_layout=grid_layout, slip_prob=slip,
        agent_type="q_learning", alpha=0.1, gamma=0.99,
        epsilon=1.0, epsilon_min=0.01,
        epsilon_decay="exponential", epsilon_decay_rate=0.99,
        num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, seeds=SEEDS,
    )
    eps_aggregated["Exponencial (rate=0.99)"] = aggregate_metrics(metrics_list)

    plot_learning_curves(
        eps_aggregated,
        title_prefix="Estrategias de Decaimento de Epsilon",
        filename=f"{OUTPUT_DIR}/hp_epsilon_comparison.png",
    )

    # --- 3d. Busca em grade (grid search) ---
    print("\n--- 3d. Busca em Grade de Hiperparametros (Q-Learning, Double Q, Expected SARSA) ---")
    search_space = {
        "alpha": [0.05, 0.1, 0.3],
        "gamma": [0.9, 0.99],
        "epsilon_min": [0.01, 0.05],
        "epsilon_decay": ["exponential"],
        "epsilon_decay_rate": [0.995, 0.99],
    }

    for agent_type, label in [
        ("q_learning", "Q-Learning"),
        ("double_q_learning", "Double Q-Learning"),
        ("expected_sarsa", "Expected SARSA"),
    ]:
        best_result, _ = grid_search_hyperparameters(
            grid_layout=grid_layout,
            slip_prob=slip,
            agent_type=agent_type,
            search_space=search_space,
            num_episodes=1200,
            max_steps=MAX_STEPS,
            seeds=SEEDS[:3],
        )

        p = best_result["params"]
        print(
            f"  {label:17s} | score={best_result['score']:.3f} | "
            f"reward={best_result['mean_reward']:.3f} | "
            f"success={best_result['success_rate']:.2%} | "
            f"steps={best_result['mean_steps']:.1f}"
        )
        print(
            f"    melhor config: alpha={p['alpha']}, gamma={p['gamma']}, "
            f"eps_min={p['epsilon_min']}, decay={p['epsilon_decay']}, "
            f"decay_rate={p['epsilon_decay_rate']}"
        )


# ============================================================
# PARTE 3E: Comparacao de Estrategias de Exploracao
# ============================================================
def run_exploration_strategy_comparison():
    print("\n" + "=" * 70)
    print("PARTE 3E: COMPARACAO DE ESTRATEGIAS DE EXPLORACAO")
    print("=" * 70)

    grid_layout = GRID_4x4
    slip = 0.2

    strategies = [
        (
            "epsilon_greedy",
            "Epsilon-Greedy",
            {
                "epsilon": 1.0,
                "epsilon_min": 0.01,
                "epsilon_decay": "exponential",
                "epsilon_decay_rate": 0.995,
            },
        ),
        (
            "softmax",
            "Softmax (Boltzmann)",
            {
                "temperature": 1.0,
                "temperature_min": 0.05,
                "temperature_decay": "exponential",
                "temperature_decay_rate": 0.995,
            },
        ),
    ]

    for agent_type, agent_label in [
        ("q_learning", "Q-Learning"),
        ("double_q_learning", "Double Q-Learning"),
        ("expected_sarsa", "Expected SARSA"),
    ]:
        print(f"\n--- {agent_label}: epsilon-greedy vs softmax ---")

        strategy_curves = {}
        strategy_eval = {}

        for strategy_key, strategy_label, strategy_cfg in strategies:
            experiment_kwargs = dict(
                grid_layout=grid_layout,
                slip_prob=slip,
                agent_type=agent_type,
                alpha=0.1,
                gamma=0.99,
                exploration_strategy=strategy_key,
                num_episodes=NUM_EPISODES,
                max_steps=MAX_STEPS,
                seeds=SEEDS,
                verbose=False,
            )
            experiment_kwargs.update(strategy_cfg)
            metrics_list, eval_list = run_experiment_multiple_seeds(**experiment_kwargs)

            strategy_curves[strategy_label] = aggregate_metrics(metrics_list)
            strategy_eval[strategy_label] = {
                "mean_reward": np.mean([e["mean_reward"] for e in eval_list]),
                "success_rate": np.mean([e["success_rate"] for e in eval_list]),
                "mean_steps": np.mean([e["mean_steps"] for e in eval_list]),
            }

            print(
                f"  {strategy_label:20s} | "
                f"Reward={strategy_eval[strategy_label]['mean_reward']:.3f} | "
                f"Success={strategy_eval[strategy_label]['success_rate']:.2%} | "
                f"Steps={strategy_eval[strategy_label]['mean_steps']:.1f}"
            )

        plot_learning_curves(
            strategy_curves,
            title_prefix=f"Comparacao de Exploracao - {agent_label}",
            filename=f"{OUTPUT_DIR}/exploration_learning_{agent_type}.png",
        )

        plot_comparison_bar(
            strategy_eval,
            filename=f"{OUTPUT_DIR}/exploration_eval_{agent_type}.png",
        )


# ============================================================
# PARTE 4: Comparacao Final VI vs Q-Learning vs Double Q-Learning
# ============================================================
def run_final_comparison():
    print("\n" + "=" * 70)
    print("PARTE 4: COMPARACAO FINAL - VI vs Q-Learning vs Double Q-Learning vs Expected SARSA")
    print("=" * 70)

    grid_layout = GRID_4x4
    slip = 0.2

    # Value Iteration (baseline otimo)
    env_vi = GridworldEnv(grid_layout=grid_layout, slip_prob=slip, seed=42)
    vi_agent = ValueIterationAgent(env_vi, gamma=0.99, theta=1e-8)
    V_vi, policy_vi, iters_vi = vi_agent.run()
    print(f"\nValue Iteration: Convergiu em {iters_vi} iteracoes")

    # Avaliar VI
    vi_eval = evaluate_policy(
        GridworldEnv(grid_layout=grid_layout, slip_prob=slip, seed=10042),
        vi_agent, num_episodes=500, max_steps=MAX_STEPS,
    )
    print(
        f"  VI Avaliacao: Reward={vi_eval['mean_reward']:.3f}, "
        f"Success={vi_eval['success_rate']:.2%}"
    )

    # Q-Learning
    q_metrics_list, q_eval_list = run_experiment_multiple_seeds(
        grid_layout=grid_layout, slip_prob=slip,
        agent_type="q_learning", alpha=0.1, gamma=0.99,
        epsilon=1.0, epsilon_min=0.01,
        epsilon_decay="exponential", epsilon_decay_rate=0.995,
        num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, seeds=SEEDS,
        verbose=True,
    )

    # Double Q-Learning
    dq_metrics_list, dq_eval_list = run_experiment_multiple_seeds(
        grid_layout=grid_layout, slip_prob=slip,
        agent_type="double_q_learning", alpha=0.1, gamma=0.99,
        epsilon=1.0, epsilon_min=0.01,
        epsilon_decay="exponential", epsilon_decay_rate=0.995,
        num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, seeds=SEEDS,
        verbose=True,
    )

    # Expected SARSA
    es_metrics_list, es_eval_list = run_experiment_multiple_seeds(
        grid_layout=grid_layout, slip_prob=slip,
        agent_type="expected_sarsa", alpha=0.1, gamma=0.99,
        epsilon=1.0, epsilon_min=0.01,
        epsilon_decay="exponential", epsilon_decay_rate=0.995,
        num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, seeds=SEEDS,
        verbose=True,
    )

    # Comparacao de avaliacao
    final_eval = {
        "Value Iteration": {
            "mean_reward": vi_eval["mean_reward"],
            "success_rate": vi_eval["success_rate"],
            "mean_steps": vi_eval["mean_steps"],
        },
        "Q-Learning": {
            "mean_reward": np.mean([e["mean_reward"] for e in q_eval_list]),
            "success_rate": np.mean([e["success_rate"] for e in q_eval_list]),
            "mean_steps": np.mean([e["mean_steps"] for e in q_eval_list]),
        },
        "Double Q-Learning": {
            "mean_reward": np.mean([e["mean_reward"] for e in dq_eval_list]),
            "success_rate": np.mean([e["success_rate"] for e in dq_eval_list]),
            "mean_steps": np.mean([e["mean_steps"] for e in dq_eval_list]),
        },
        "Expected SARSA": {
            "mean_reward": np.mean([e["mean_reward"] for e in es_eval_list]),
            "success_rate": np.mean([e["success_rate"] for e in es_eval_list]),
            "mean_steps": np.mean([e["mean_steps"] for e in es_eval_list]),
        },
    }

    print("\n--- Resultados Finais de Avaliacao ---")
    for name, vals in final_eval.items():
        print(
            f"  {name:20s}: Reward={vals['mean_reward']:.3f}, "
            f"Success={vals['success_rate']:.2%}, "
            f"Steps={vals['mean_steps']:.1f}"
        )

    plot_comparison_bar(
        final_eval,
        filename=f"{OUTPUT_DIR}/final_comparison_bar.png",
    )

    # Curvas de aprendizado Q vs Double Q
    plot_learning_curves(
        {
            "Q-Learning": aggregate_metrics(q_metrics_list),
            "Double Q-Learning": aggregate_metrics(dq_metrics_list),
            "Expected SARSA": aggregate_metrics(es_metrics_list),
        },
        title_prefix="Q-Learning vs Double Q-Learning vs Expected SARSA (4x4, slip=0.2)",
        filename=f"{OUTPUT_DIR}/final_learning_curves.png",
    )

    # Plotar VI como referencia
    plot_value_and_policy(
        env_vi, V_vi, policy_vi,
        suptitle="Value Iteration - Solucao Otima (gamma=0.99, slip=0.2)",
        filename=f"{OUTPUT_DIR}/vi_optimal_solution.png",
    )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    ensure_output_dir()

    print("Gridworld com Perigos - Experimentos de Aprendizado por Reforco")
    print("MC906 - Unicamp")
    print("=" * 70)

    # Executar todas as partes
    run_value_iteration_experiments()
    run_qlearning_experiments()
    run_hyperparameter_study()
    run_exploration_strategy_comparison()
    run_final_comparison()

    print("\n" + "=" * 70)
    print("EXPERIMENTOS CONCLUIDOS!")
    print(f"Resultados salvos em: {OUTPUT_DIR}/")
    print("=" * 70)
