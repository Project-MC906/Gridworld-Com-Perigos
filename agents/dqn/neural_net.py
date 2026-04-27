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

    def __init__(self, in_features, out_features, xp=np, seed=None):
        self.xp = xp
        rng = np.random.default_rng(seed)
        # Inicializacao He (adequada para ReLU)
        std = np.sqrt(2.0 / in_features)
        weights = rng.normal(0.0, std, (out_features, in_features)).astype(np.float32)
        self.W = self.xp.asarray(weights)
        self.b = self.xp.zeros(out_features, dtype=self.xp.float32)

        # Gradientes
        self.dW = self.xp.zeros_like(self.W)
        self.db = self.xp.zeros_like(self.b)

        # Cache para backprop
        self._input = None

    def forward(self, x):
        x = x.astype(self.W.dtype, copy=False)
        self._input = x
        if x.ndim == 1:
            return self.W @ x + self.b
        return x @ self.W.T + self.b

    def backward(self, grad_out):
        """Retorna gradiente em relacao a entrada."""
        if grad_out.ndim == 1:
            self.dW = self.xp.outer(grad_out, self._input)
            self.db = grad_out.copy()
            return self.W.T @ grad_out

        self.dW = grad_out.T @ self._input
        self.db = self.xp.sum(grad_out, axis=0)
        return grad_out @ self.W

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
    xp : module
        Backend numerico (numpy ou cupy).
    seed : int or None
    """

    def __init__(self, input_size, hidden_sizes, output_size, xp=np, seed=42):
        self.xp = xp
        self.layers = []
        rng = np.random.default_rng(seed)
        sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(sizes) - 1):
            layer_seed = int(rng.integers(0, 2**31))
            self.layers.append(Linear(sizes[i], sizes[i + 1], xp=self.xp, seed=layer_seed))
            # Adiciona ReLU em todas as camadas exceto a ultima
            if i < len(sizes) - 2:
                self.layers.append(ReLU())

        # Otimizador Adam para cada parametro
        self._adam_states = {}
        self._t = 0  # passo global do Adam

    def forward(self, x):
        """Passagem para frente. x pode ser 1D (estado) ou 2D (batch)."""
        out = self.xp.asarray(x, dtype=self.xp.float32)
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
        loss = self.xp.mean(diff ** 2)
        grad = 2.0 * diff / diff.size
        return loss, grad

    def zero_grad(self):
        """Zera gradientes acumulados das camadas lineares."""
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.dW[...] = 0.0
                layer.db[...] = 0.0

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
                        "m": self.xp.zeros_like(param),
                        "v": self.xp.zeros_like(param),
                    }
                state = self._adam_states[pid]
                state["m"] = beta1 * state["m"] + (1 - beta1) * grad
                state["v"] = beta2 * state["v"] + (1 - beta2) * grad ** 2
                m_hat = state["m"] / (1 - beta1 ** self._t)
                v_hat = state["v"] / (1 - beta2 ** self._t)
                param -= lr * m_hat / (self.xp.sqrt(v_hat) + eps)

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
                layer.W[:] = self.xp.asarray(weights[idx][0])
                layer.b[:] = self.xp.asarray(weights[idx][1])
                idx += 1
