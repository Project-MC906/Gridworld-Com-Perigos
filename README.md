# Gridworld com Perigos - Aprendizado por Reforco

MC906 - Inteligencia Artificial - Unicamp

## Requisitos

- Python 3.8+
- NumPy
- Matplotlib

Instalar dependencias:

```bash
pip install numpy matplotlib
```

## Como executar

```bash
python main.py
```

Os graficos serao salvos na pasta `resultados/`.

## Estrutura dos arquivos

| Arquivo | Descricao |
|---------|-----------|
| `gridworld_env.py` | Ambiente Gridworld (MDP, transicoes estocasticas, recompensas) |
| `value_iteration.py` | Value Iteration (Equacoes de Bellman) |
| `q_learning.py` | Q-Learning, Double Q-Learning e Expected SARSA com exploracao epsilon-greedy e softmax |
| `experiment.py` | Runner experimental com metricas e multiplas seeds |
| `visualization.py` | Heatmaps, setas de politica, curvas de aprendizado, trajetorias |
| `main.py` | Script principal que executa todos os experimentos |

## Comparacao de exploracao

O script principal gera comparacoes entre estrategias de exploracao para
Q-Learning, Double Q-Learning e Expected SARSA:

- Epsilon-greedy (com decaimento)
- Softmax/Boltzmann (com decaimento de temperatura)

Arquivos de saida esperados:

- `resultados/exploration_learning_q_learning.png`
- `resultados/exploration_learning_double_q_learning.png`
- `resultados/exploration_learning_expected_sarsa.png`
- `resultados/exploration_eval_q_learning.png`
- `resultados/exploration_eval_double_q_learning.png`
- `resultados/exploration_eval_expected_sarsa.png`
