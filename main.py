from losses import dirichlet_energy_loss as loss_function

import plotting
import numpy as np
import torch

if __name__ == "__main__":
    torch.manual_seed(0)
    analytical_sol = lambda points: torch.sin(torch.pi * points[:, 0]) * torch.sin(2*torch.pi * points[:, 1])
    plotting.plot_fn(analytical_sol, 200)

