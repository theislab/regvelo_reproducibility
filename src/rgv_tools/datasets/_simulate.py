import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform


# Adapted from https://github.com/theislab/scvelo/blob/22b6e7e6cdb3c321c5a1be4ab2f29486ba01ab4f/scvelo/datasets/_simulate.py#L77
def draw_poisson(n: int, seed: int) -> Tensor:
    """TODO."""
    torch.random.manual_seed(seed)

    distribution = Uniform(low=torch.finfo(torch.float32).eps, high=1)
    t = torch.cumsum(-0.1 * distribution.sample(sample_shape=torch.Size([n - 1])).log(), dim=0)
    return torch.cat([torch.zeros(size=(1,)), t], dim=0)


def get_sde_parameters(n_obs: int, n_vars: int, seed: int):
    mu = torch.tensor([5, 0.5, 0.125], dtype=torch.float32).log()
    R = torch.tensor([[1.0, 0.2, 0.2], [0.2, 1.0, 0.8], [0.2, 0.8, 1.0]], dtype=torch.float32)
    C = torch.tensor([0.4, 0.4, 0.4], dtype=torch.float32)[:, None]
    cov = C * C.T * R
    distribution = MultivariateNormal(loc=mu, covariance_matrix=cov)
    alpha, beta, gamma = distribution.sample(sample_shape=torch.Size([n_vars])).exp().T

    mean_alpha = alpha.mean()
    coef_m = torch.tensor(
        [
            [0, 1, -mean_alpha, 2, 2],
            [1, 0, -mean_alpha, 2, 2],
            [0, 2, mean_alpha, 2, 4],
            [0, 3, mean_alpha, 2, 4],
            [2, 3, -mean_alpha, 2, 2],
            [3, 2, -mean_alpha, 2, 2],
            [1, 4, mean_alpha, 2, 4],
            [1, 5, mean_alpha, 2, 4],
            [4, 5, -mean_alpha, 2, 2],
            [5, 4, -mean_alpha, 2, 2],
        ]
    )

    n_regulators = 6
    n_targets = 6
    K = torch.zeros([n_targets, n_regulators], dtype=torch.float32)
    n = torch.zeros([n_targets, n_regulators], dtype=torch.float32)
    h = torch.zeros([n_targets, n_regulators], dtype=torch.float32)

    K[coef_m[:, 1].int(), coef_m[:, 0].int()] = coef_m[:, 2]
    n[coef_m[:, 1].int(), coef_m[:, 0].int()] = coef_m[:, 3]
    h[coef_m[:, 1].int(), coef_m[:, 0].int()] = coef_m[:, 4]

    t = draw_poisson(n_obs, seed=seed)

    return K, n, h, beta, gamma, t


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
