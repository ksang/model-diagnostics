import torch
from torch.autograd import Variable
from typing import Union, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def initialize_inputs_baseline(
    inputs: Union[torch.Tensor, List[torch.Tensor]],
    baseline: Union[torch.Tensor, List[torch.Tensor]],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    xs = inputs
    bs = baseline
    if baseline is None:
        if torch.is_tensor(inputs):
            bs = torch.zeros(inputs.shape)
        else:
            bs = []
            for x in inputs:
                bs.append(torch.zeros(x.shape))

    if torch.is_tensor(inputs):
        assert xs.shape == bs.shape, "input baseline shape not equal"
        xs = [xs]
        bs = [bs]
    batch_size = xs[0].shape[0]
    for i, x in enumerate(xs):
        assert x.shape[0] == batch_size, "input batch size not equal"
        assert x.shape == bs[i].shape, "input baseline shape not equal"
        if torch.is_floating_point(x):
            xs[i] = Variable(x, requires_grad=True)
    return xs, bs


def plot_vector_gradients(tokens, gradients, title, colormap="GnBu"):
    fig, ax = plt.subplots(figsize=(21, 3))
    xvals = [x + str(i) for i, x in enumerate(tokens)]
    norm_grad = (gradients - np.min(gradients)) / (
        np.max(gradients) - np.min(gradients)
    )
    cmap = plt.colormaps[colormap]
    plt.tick_params(axis="both", which="minor", labelsize=29)
    p = plt.bar(xvals, gradients, color=cmap(norm_grad), linewidth=1)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
        ax=ax,
        label="Normalized Color Map",
    )
    plt.title(title)
    p = plt.xticks(
        ticks=[i for i in range(len(tokens))], labels=tokens, fontsize=12, rotation=0
    )
