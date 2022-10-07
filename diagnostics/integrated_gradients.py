import numpy as np
import torch
import torch.nn as nn
from typing import Union, List
from torch.autograd import Variable

from diagnostics.utils import initialize_inputs_baseline


def compute_gradients(
    model: nn.Module,
    scaled_inputs: List[List[torch.Tensor]],
    baseline: List[torch.Tensor],
    target: torch.Tensor,
) -> List[List[torch.Tensor]]:
    grads = [[] for _ in range(len(baseline))]
    for inputs in scaled_inputs:
        pred = model(*inputs)
        base_out = model(*baseline)
        batch_size = pred.shape[0]
        diff = torch.sum((pred * target - base_out).view(batch_size, -1), -1)
        model.zero_grad()
        diff.sum().backward()
        for i, x in enumerate(inputs):
            grads[i].append(x.grad.unsqueeze(0))
    return grads


def get_integrated_gradients(
    model: nn.Module,
    inputs: Union[torch.Tensor, List[torch.Tensor]],
    baseline: Union[torch.Tensor, List[torch.Tensor]],
    target: torch.Tensor,
    steps: int = 50,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Get integrated gradients with respect to the model output compare to baseline.
    "Axiomatic Attribution for Deep Networks", Mukund Sundararajan, Ankur Taly, Qiqi Yan
    https://arxiv.org/abs/1703.01365
    Args:
        model (nn.Module): Targeted pytorch model.
        inputs (torch.Tensor, List[torch.Tensor]): Inputs of the model,
            can be single tensor or list of tensors, expected shape BxD or BxCxHxW.
        baseline (torch.Tensor, List[torch.Tensor]): Baseline input, used to feed into model
            get output for comparison and backprogate gradients, expected shape same as inputs.
        target (torch.Tensor): Target label index for the batch,
            for classification task, this is the class id, expected shape one hot tensor BxD.
        steps (optional: int): Number of steps used to approximate compute integral.
    Return:
        integrated_gradients (torch.Tensor, List[torch.Tensor])
    """
    xs, bs = initialize_inputs_baseline(inputs, baseline)
    # Riemman approximation of integral
    scaled_inputs = []
    for i in range(0, steps + 1):
        sinputs = [bs[j] + (float(i) / steps) * (x - bs[j]) for j, x in enumerate(xs)]
        sinputs = [Variable(x, requires_grad=True) for x in sinputs]
        scaled_inputs.append(sinputs)
    grads = compute_gradients(model, scaled_inputs, bs, target)
    # Use trapezoidal rule to approximate the integral.
    # See Section 4 of the following paper for an accuracy comparison between
    # left, right, and trapezoidal IG approximations:
    # "Computing Linear Restrictions of Neural Networks", Matthew Sotoudeh, Aditya V. Thakur
    # https://arxiv.org/abs/1908.06214
    avg_grads = []
    for i, grad in enumerate(grads):
        grad = torch.cat((grad[:-1] + grad[1:])) / 2.0
        avg_grads.append(torch.mean(grad, axis=0))

    integrated_grad = []
    for i, x in enumerate(xs):
        delta_X = (x - bs[i]).detach()
        integrated_grad.append(delta_X * avg_grads[i])
    if torch.is_tensor(inputs):
        return integrated_grad[0]
    return integrated_grad
