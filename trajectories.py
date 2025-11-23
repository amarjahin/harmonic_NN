import torch
import numpy as np

def ho_trajectory(x0, v0, t, omega):
    """
    Analytic solution for undamped harmonic oscillator.
    x0, v0, t can be torch tensors of same shape.
    """
    x = x0 * torch.cos(omega * t) + (v0 / omega) * torch.sin(omega * t)
    v = -x0 * omega * torch.sin(omega * t) + v0 * torch.cos(omega * t)
    return x, v

def generate_dataset(
    n_samples=50_000,
    t_min=0.0,
    t_max=10.0,
    x0_range=(-1.0, 1.0),
    v0_range=(-1.0, 1.0),
    omega=2*np.pi,
    device="cpu",
):
    """
    Generate (x0, v0, t) -> (x(t), v(t)) pairs.
    """
    x0 = (x0_range[1] - x0_range[0]) * torch.rand(n_samples, device=device) + x0_range[0]
    v0 = (v0_range[1] - v0_range[0]) * torch.rand(n_samples, device=device) + v0_range[0]
    t = (t_max - t_min) * torch.rand(n_samples, device=device) + t_min

    x, v = ho_trajectory(x0, v0, t, omega=omega)

    # Inputs: [x0, v0, t], Targets: [x(t), v(t)]
    X = torch.stack([x0, v0, t], dim=1)
    y = torch.stack([x, v], dim=1)
    return X, y