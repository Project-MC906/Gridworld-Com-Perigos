"""
Modulo de visualizacao grafica.

Gera:
1. Mapa de calor da funcao de valor V(s)
2. Mapa de setas da politica derivada pi(s)
3. Curvas de aprendizado (recompensa, passos, taxa de sucesso)
4. Curva de convergencia de Value Iteration
5. Trajetoria do agente no grid
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from gridworld_env import (
    WALL, TRAP, GOAL, START, EMPTY,
    UP, RIGHT, DOWN, LEFT, ACTION_NAMES, NUM_ACTIONS,
)

# Direcao das setas para cada acao
ARROW_DX = {UP: 0, RIGHT: 0.3, DOWN: 0, LEFT: -0.3}
ARROW_DY = {UP: 0.3, RIGHT: 0, DOWN: -0.3, LEFT: 0}


def _get_value_grid(env, V):
    """Converte vetor de valores para uma matriz 2D do grid."""
    value_grid = np.full((env.rows, env.cols), np.nan)
    for s_idx, (r, c) in enumerate(env.states):
        value_grid[r, c] = V[s_idx]
    return value_grid


def _get_policy_grid(env, policy):
    """Converte vetor de politica para uma matriz 2D do grid."""
    policy_grid = np.full((env.rows, env.cols), -1, dtype=int)
    for s_idx, (r, c) in enumerate(env.states):
        policy_grid[r, c] = policy[s_idx]
    return policy_grid


def plot_value_heatmap(env, V, title="Funcao de Valor V(s)", filename=None, ax=None):
    """
    Plota mapa de calor da funcao de valor V(s).

    Parametros
    ----------
    env : GridworldEnv
    V : np.ndarray
        Vetor de valores por estado.
    title : str
    filename : str or None
        Se fornecido, salva a figura.
    ax : matplotlib.axes.Axes or None
        Se fornecido, plota no axes dado.
    """
    value_grid = _get_value_grid(env, V)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(6, env.cols), max(5, env.rows)))

    # Criar mascara para paredes
    masked_grid = np.ma.array(value_grid, mask=np.isnan(value_grid))

    im = ax.imshow(masked_grid, cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, label="V(s)")

    # Anotar valores e tipos de celula
    for r in range(env.rows):
        for c in range(env.cols):
            cell = env.grid[r, c]
            if cell == WALL:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           fill=True, color="gray", alpha=0.8))
                ax.text(c, r, "W", ha="center", va="center",
                        fontsize=12, fontweight="bold", color="white")
            elif cell == TRAP:
                ax.text(c, r - 0.15, "TRAP", ha="center", va="center",
                        fontsize=7, fontweight="bold", color="red")
                ax.text(c, r + 0.2, f"{value_grid[r, c]:.2f}", ha="center",
                        va="center", fontsize=8)
            elif cell == GOAL:
                ax.text(c, r - 0.15, "GOAL", ha="center", va="center",
                        fontsize=7, fontweight="bold", color="darkgreen")
                ax.text(c, r + 0.2, f"{value_grid[r, c]:.2f}", ha="center",
                        va="center", fontsize=8)
            elif cell == START:
                ax.text(c, r - 0.15, "S", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="blue")
                ax.text(c, r + 0.2, f"{value_grid[r, c]:.2f}", ha="center",
                        va="center", fontsize=8)
            else:
                if not np.isnan(value_grid[r, c]):
                    ax.text(c, r, f"{value_grid[r, c]:.2f}", ha="center",
                            va="center", fontsize=8)

    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linewidth=0.5, alpha=0.3)

    if own_fig:
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.show()


def plot_policy_arrows(env, policy, V=None, title="Politica pi(s)", filename=None, ax=None):
    """
    Plota mapa de setas da politica derivada.

    Parametros
    ----------
    env : GridworldEnv
    policy : np.ndarray
        Vetor de acoes por estado.
    V : np.ndarray or None
        Se fornecido, usa como fundo de calor.
    title : str
    filename : str or None
    ax : matplotlib.axes.Axes or None
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(6, env.cols), max(5, env.rows)))

    # Fundo de calor se V disponivel
    if V is not None:
        value_grid = _get_value_grid(env, V)
        masked_grid = np.ma.array(value_grid, mask=np.isnan(value_grid))
        ax.imshow(masked_grid, cmap="RdYlGn", interpolation="nearest", alpha=0.4)

    # Desenhar setas e anotacoes
    for r in range(env.rows):
        for c in range(env.cols):
            cell = env.grid[r, c]
            if cell == WALL:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           fill=True, color="gray", alpha=0.8))
                ax.text(c, r, "W", ha="center", va="center",
                        fontsize=12, fontweight="bold", color="white")
            elif cell == TRAP:
                ax.text(c, r, "T", ha="center", va="center",
                        fontsize=14, fontweight="bold", color="red")
            elif cell == GOAL:
                ax.text(c, r, "G", ha="center", va="center",
                        fontsize=14, fontweight="bold", color="darkgreen")
            else:
                if (r, c) in env.state_to_idx:
                    s_idx = env.state_to_idx[(r, c)]
                    action = policy[s_idx]
                    dx = ARROW_DX[action]
                    dy = ARROW_DY[action]
                    # Nota: no imshow, eixo y eh invertido
                    ax.annotate(
                        "",
                        xy=(c + dx, r - dy),
                        xytext=(c, r),
                        arrowprops=dict(
                            arrowstyle="->",
                            color="black",
                            lw=2,
                        ),
                    )
                    if cell == START:
                        ax.text(c, r + 0.35, "S", ha="center", va="center",
                                fontsize=7, color="blue", fontweight="bold")

    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.set_aspect("equal")

    if own_fig:
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.show()


def plot_value_and_policy(env, V, policy, suptitle="", filename=None):
    """Plota mapa de calor e setas lado a lado."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, env.cols * 2 + 2), max(5, env.rows)))
    plot_value_heatmap(env, V, title="Funcao de Valor V(s)", ax=ax1)
    plot_policy_arrows(env, policy, V=V, title="Politica pi(s)", ax=ax2)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


def smooth_curve(data, window=50):
    """Suaviza dados com media movel exponencial (EMA)."""
    ema = np.zeros_like(data, dtype=float)
    alpha = 2.0 / (window + 1)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def plot_learning_curves(
    metrics_dict,
    title_prefix="",
    window=50,
    filename=None,
):
    """
    Plota curvas de aprendizado para multiplos agentes/configuracoes.

    Parametros
    ----------
    metrics_dict : dict
        Dicionario {nome: aggregated_metrics} onde aggregated_metrics contem
        'rewards_mean', 'rewards_std', 'steps_mean', 'successes_mean', etc.
    title_prefix : str
    window : int
        Janela de suavizacao EMA.
    filename : str or None
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_dict)))

    for idx, (name, agg) in enumerate(metrics_dict.items()):
        color = colors[idx]
        episodes = np.arange(1, agg["num_episodes"] + 1)

        # 1. Recompensa acumulada
        ax = axes[0, 0]
        smoothed = smooth_curve(agg["rewards_mean"], window)
        ax.plot(episodes, smoothed, label=name, color=color, linewidth=1.5)
        ax.fill_between(
            episodes,
            smooth_curve(agg["rewards_mean"] - agg["rewards_std"], window),
            smooth_curve(agg["rewards_mean"] + agg["rewards_std"], window),
            alpha=0.15, color=color,
        )
        ax.set_xlabel("Episodio")
        ax.set_ylabel("Recompensa Acumulada")
        ax.set_title("Recompensa por Episodio (EMA)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2. Passos por episodio
        ax = axes[0, 1]
        smoothed = smooth_curve(agg["steps_mean"], window)
        ax.plot(episodes, smoothed, label=name, color=color, linewidth=1.5)
        ax.set_xlabel("Episodio")
        ax.set_ylabel("Passos")
        ax.set_title("Duracao do Episodio (EMA)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. Taxa de sucesso (janela deslizante)
        ax = axes[1, 0]
        success_smooth = smooth_curve(agg["successes_mean"], window)
        ax.plot(episodes, success_smooth, label=name, color=color, linewidth=1.5)
        ax.set_xlabel("Episodio")
        ax.set_ylabel("Taxa de Sucesso")
        ax.set_title("Taxa de Sucesso (EMA)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 4. TD Error medio
        ax = axes[1, 1]
        smoothed = smooth_curve(agg["td_errors_mean"], window)
        ax.plot(episodes, smoothed, label=name, color=color, linewidth=1.5)
        ax.set_xlabel("Episodio")
        ax.set_ylabel("|TD Error| medio")
        ax.set_title("Erro de Diferenca Temporal (EMA)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


def plot_convergence_vi(delta_history, title="Convergencia de Value Iteration", filename=None):
    """Plota a curva de convergencia (delta por iteracao) de Value Iteration."""
    fig, ax = plt.subplots(figsize=(8, 5))
    iterations = np.arange(1, len(delta_history) + 1)
    ax.semilogy(iterations, delta_history, color="darkblue", linewidth=1.5)
    ax.set_xlabel("Iteracao")
    ax.set_ylabel("Delta (max |V_new - V_old|)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


def plot_trajectory(env, policy, title="Trajetoria do Agente", max_steps=100, filename=None):
    """
    Plota a trajetoria do agente seguindo a politica gulosa sobre o grid.
    """
    fig, ax = plt.subplots(figsize=(max(6, env.cols), max(5, env.rows)))

    # Desenhar grid de fundo
    grid_display = np.ones((env.rows, env.cols, 3))  # branco
    for r in range(env.rows):
        for c in range(env.cols):
            cell = env.grid[r, c]
            if cell == WALL:
                grid_display[r, c] = [0.3, 0.3, 0.3]  # cinza escuro
            elif cell == TRAP:
                grid_display[r, c] = [1.0, 0.6, 0.6]  # vermelho claro
            elif cell == GOAL:
                grid_display[r, c] = [0.6, 1.0, 0.6]  # verde claro
            elif cell == START:
                grid_display[r, c] = [0.6, 0.6, 1.0]  # azul claro

    ax.imshow(grid_display, interpolation="nearest")

    # Rotulos
    for r in range(env.rows):
        for c in range(env.cols):
            cell = env.grid[r, c]
            if cell == WALL:
                ax.text(c, r, "W", ha="center", va="center", fontsize=12,
                        fontweight="bold", color="white")
            elif cell == TRAP:
                ax.text(c, r, "T", ha="center", va="center", fontsize=12,
                        fontweight="bold", color="red")
            elif cell == GOAL:
                ax.text(c, r, "G", ha="center", va="center", fontsize=12,
                        fontweight="bold", color="darkgreen")
            elif cell == START:
                ax.text(c, r, "S", ha="center", va="center", fontsize=12,
                        fontweight="bold", color="darkblue")

    # Simular trajetoria deterministica
    state_idx = env.state_to_idx[env.start_pos]
    trajectory = [env.start_pos]
    visited = set()
    visited.add(state_idx)

    for _ in range(max_steps):
        pos = env.states[state_idx]
        if env._is_terminal(pos):
            break
        action = policy[state_idx]
        next_pos = env._get_next_pos(pos[0], pos[1], action)
        next_idx = env.state_to_idx[next_pos]
        trajectory.append(next_pos)
        if next_idx in visited:
            break  # evitar loop infinito
        visited.add(next_idx)
        state_idx = next_idx

    # Desenhar trajetoria
    if len(trajectory) > 1:
        rows_path = [pos[0] for pos in trajectory]
        cols_path = [pos[1] for pos in trajectory]
        ax.plot(cols_path, rows_path, "o-", color="darkorange", linewidth=2,
                markersize=6, markerfacecolor="orange", markeredgecolor="darkorange",
                zorder=5)
        # Marca inicio e fim
        ax.plot(cols_path[0], rows_path[0], "s", color="blue", markersize=12, zorder=6)
        ax.plot(cols_path[-1], rows_path[-1], "*", color="gold", markersize=16, zorder=6)

    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


def plot_comparison_bar(eval_results_dict, filename=None):
    """
    Plota grafico de barras comparando metricas de avaliacao entre algoritmos.

    Parametros
    ----------
    eval_results_dict : dict
        {nome_algoritmo: {'mean_reward': ..., 'success_rate': ..., 'mean_steps': ...}}
    """
    names = list(eval_results_dict.keys())
    metrics_to_plot = [
        ("mean_reward", "Recompensa Media"),
        ("success_rate", "Taxa de Sucesso"),
        ("mean_steps", "Passos Medios"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    for ax, (metric_key, metric_label) in zip(axes, metrics_to_plot):
        values = [eval_results_dict[name][metric_key] for name in names]
        bars = ax.bar(names, values, color=colors[:len(names)], edgecolor="black", alpha=0.8)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(True, axis="y", alpha=0.3)

        # Anotar valores nas barras
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    plt.suptitle("Comparacao de Desempenho entre Algoritmos", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
