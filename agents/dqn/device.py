"""
Utilitarios de backend para executar o DQN em CPU (NumPy) ou CUDA (CuPy).
"""

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover - ambiente sem CUDA/CuPy
    cp = None


def get_array_module(device="auto"):
    """
    Resolve backend numerico a partir do device solicitado.

    Parametros
    ----------
    device : str
        "auto", "cpu" ou "cuda".

    Retorna
    -------
    xp : module
        numpy ou cupy.
    resolved_device : str
        "cpu" ou "cuda".
    """
    requested = (device or "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device invalido. Use 'auto', 'cpu' ou 'cuda'.")

    if requested == "cpu":
        return np, "cpu"

    if cp is None:
        if requested == "cuda":
            raise RuntimeError(
                "CuPy nao encontrado. Instale CuPy para usar device='cuda'."
            )
        return np, "cpu"

    try:
        num_devices = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        if requested == "cuda":
            raise RuntimeError("CUDA indisponivel no ambiente atual.") from exc
        return np, "cpu"

    if num_devices < 1:
        if requested == "cuda":
            raise RuntimeError("Nenhuma GPU CUDA encontrada.")
        return np, "cpu"

    return cp, "cuda"


def to_numpy(array):
    """Converte array NumPy/CuPy para NumPy."""
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)
