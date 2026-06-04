import matplotlib.pyplot as plt
import torch

from matplotlib import cm

def plot_fn(fn, res, title=None):
    x = torch.linspace(0, 1, res)
    y = torch.linspace(0, 1, res)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    f = fn(points).detach().numpy().reshape(res, res)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(grid_x, grid_y, f, cmap=cm.Blues)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

    plt.show()