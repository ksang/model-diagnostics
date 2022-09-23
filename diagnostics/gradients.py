import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import Union, List


def get_gradients(
    model: nn.Module,
    inputs: Union[torch.Tensor, List[torch.Tensor]],
    baseline: torch.Tensor,
    target: torch.Tensor,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Get input gradients with respect to the model output compare to baseline.
    Args:
        model (nn.Module): Targeted pytorch model.
        inputs (torch.Tensor, List[torch.Tensor]): Inputs of the model,
            can be single tensor or list of tensors, expected shape BxD or BxCxHxW.
        baseline (torch.Tensor): Baseline output, used to compare
            with model output and backprogate gradients, expected shape BxD.
        target (orch.Tensor): Target label index for the batch,
            for classification task, this is the class id, expected shape one hot tensor BxD.
    Return:
        gradients (torch.Tensor, List[torch.Tensor])
    """
    xs = inputs
    if torch.is_tensor(inputs):
        xs = [xs]
    batch_size = xs[0].shape[0]
    for i, x in enumerate(xs):
        assert x.shape[0] == batch_size, "input batch size not equal"
        xs[i] = Variable(x, requires_grad=True)

    pred = model(*xs)
    assert pred.shape == baseline.shape, "output shape not equal to baseline"
    assert target.shape == pred.shape, "target shape not equal to output"
    # diff = pred * target - baseline
    output_dim = len(pred.shape) - 1
    diff = torch.sum(pred * target - baseline, -1 * output_dim)
    diff.backward()

    if torch.is_tensor(inputs):
        return xs[0].grad
    res = []
    for x in xs:
        res.append(x.grad)
    return res
