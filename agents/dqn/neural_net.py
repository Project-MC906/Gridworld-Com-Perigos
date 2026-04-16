"""
Rede neural densa implementada do zero com NumPy.

Suporta:
- Camadas densas (Linear)
- Ativacoes: ReLU, Linear (identidade na saida)
- Loss: MSE
- Otimizador: Adam
- Backpropagation completo

Nao usa PyTorch, TensorFlow ou qualquer lib de deep learning.
"""

import numpy as np


class Linear:
    """Camada densa: y = Wx + b"""

    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.default_rng(seed)
        # Inicializacao He (adequada para ReLU)
        std = np.sqrt(2.0 / in_features)
        self.W = rng.normal(0.0, std, (out_features, in_features))
        self.b = np.zeros(out_features)

        # Gradientes
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Cache para backprop
        self._input = None

    def forward(self, x):
        self._input = x
        return self.W @ x + self.b

    def backward(self, grad_out):
        """Retorna gradiente em relacao a entrada."""
        self.dW = np.outer(grad_out, self._input)
        self.db = grad_out.copy()
        return self.W.T @ grad_out

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    """Ativacao ReLU: f(x) = max(0, x)"""

    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_out):
        return grad_out * self._mask


class NeuralNetwork:
    """
    Rede neural densa sequencial.

    Arquitetura padrao para DQN em Gridworld:
        input -> Linear(in, h) -> ReLU -> Linear(h, h) -> ReLU -> Linear(h, out)

    Parametros
    ----------
    input_size : int
        Dimensao da entrada (normalmente one-hot do estado ou tamanho do estado).
    hidden_sizes : list of int
        Tamanhos das camadas ocultas.
    output_size : int
        Numero de acoes (saidas Q(s,a)).
    seed : int or None
    """

    def __init__(self, input_size, hidden_sizes, output_size, seed=42):
        self.layers = []
        rng = np.random.default_rng(seed)
        sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(sizes) - 1):
            layer_seed = int(rng.integers(0, 2**31))
            self.layers.append(Linear(sizes[i], sizes[i + 1], seed=layer_seed))
            # Adiciona ReLU em todas as camadas exceto a ultima
            if i < len(sizes) - 2:
                self.layers.append(ReLU())

        # Otimizador Adam para cada parametro
        self._adam_states = {}
        self._t = 0  # passo global do Adam

    def forward(self, x):
        """Passagem para frente. x: vetor 1D (input_size,)"""
        out = x.astype(float)
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_loss):
        """Backpropagation. grad_loss: gradiente da loss em relacao a saida."""
        grad = grad_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def mse_loss_and_grad(self, predictions, targets):
        """
        Calcula MSE loss e gradiente em relacao as predicoes.

        loss = mean((pred - target)^2)
        d_loss/d_pred = 2*(pred - target) / N
        """
        diff = predictions - targets
        loss = np.mean(diff ** 2)
        grad = 2.0 * diff / len(diff)
        return loss, grad

    def update_adam(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        """Atualiza pesos com otimizador Adam."""
        self._t += 1
        for layer in self.layers:
            if not isinstance(layer, Linear):
                continue
            for param, grad in layer.params():
                # Identificador unico para cada array de parametro
                pid = id(param)
                if pid not in self._adam_states:
                    self._adam_states[pid] = {
                        "m": np.zeros_like(param),
                        "v": np.zeros_like(param),
                    }
                state = self._adam_states[pid]
                state["m"] = beta1 * state["m"] + (1 - beta1) * grad
                state["v"] = beta2 * state["v"] + (1 - beta2) * grad ** 2
                m_hat = state["m"] / (1 - beta1 ** self._t)
                v_hat = state["v"] / (1 - beta2 ** self._t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def get_weights(self):
        """Retorna copia de todos os pesos para uso na target network."""
        weights = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                weights.append((layer.W.copy(), layer.b.copy()))
        return weights

    def set_weights(self, weights):
        """Carrega pesos (usado para sincronizar target network)."""
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W[:] = weights[idx][0]
                layer.b[:] = weights[idx][1]
                idx += 1
