import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor


# Adapted from https://github.com/theislab/scvelo/blob/22b6e7e6cdb3c321c5a1be4ab2f29486ba01ab4f/scvelo/datasets/_simulate.py#L77
def draw_poisson(n: int, random_seed: int) -> ArrayLike:
    """TODO."""
    from random import seed, uniform

    seed(random_seed)
    t = np.cumsum([-0.1 * np.log(uniform(0, 1)) for _ in range(n - 1)])
    return np.insert(t, 0, 0)


class VelocityEncoder(torch.nn.Module):
    """TODO."""

    noise_type = "scalar"
    sde_type = "ito"

    def __init__(self, K: Tensor, n: Tensor, h: Tensor, alpha_b: Tensor, beta: Tensor, gamma: Tensor):
        """TODO."""
        super().__init__()
        self.K = K
        self.n = n
        self.h = h
        self.alpha_b = alpha_b
        self.beta = beta
        self.gamma = gamma

    # Drift
    def f(self, t: Tensor, y: Tensor) -> Tensor:
        """TODO."""
        y = y.T
        u = y[0 : int(y.shape[0] / 2), 0].ravel()
        s = y[int(y.shape[0] / 2) :, 0].ravel()

        sign = torch.sign(self.K)
        sign = torch.clip(torch.sign(self.K), 0, 1)

        s_m = s.repeat(self.n.shape[0], 1)
        x_n = torch.pow(
            torch.clip(
                s_m,
                0,
            ),
            self.n,
        )
        h_n = self.h**self.n

        p_act = x_n / (h_n + x_n)
        p_neg = h_n / (h_n + x_n)

        p = torch.abs(self.K) * ((p_act * sign) + (p_neg * (1 - sign)))
        alpha = p.sum(1) + self.alpha_b

        du = (
            torch.clip(
                alpha,
                0,
            )
            - self.beta * u
        )
        ds = self.beta * u - self.gamma * s

        du = du.reshape((-1, 1))
        ds = ds.reshape((-1, 1))

        v = torch.concatenate([du, ds]).reshape(1, -1)

        return v

    # Diffusion
    def g(self, t: Tensor, y: Tensor) -> Tensor:
        """TODO."""
        return 0.1 * torch.randn([1, y.shape[1]]).view(1, y.shape[1], 1)
