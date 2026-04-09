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
| `q_learning.py` | Q-Learning e Double Q-Learning |
| `experiment.py` | Runner experimental com metricas e multiplas seeds |
| `visualization.py` | Heatmaps, setas de politica, curvas de aprendizado, trajetorias |
| `main.py` | Script principal que executa todos os experimentos |
