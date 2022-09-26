import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import Union, List


def get_gradients(
    model: nn.Module,
    inputs: Union[torch.Tensor, List[torch.Tensor]],
    baseline: Union[torch.Tensor, List[torch.Tensor]],
    target: torch.Tensor,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Get input gradients with respect to the model output compare to baseline.
    Args:
        model (nn.Module): Targeted pytorch model.
        inputs (torch.Tensor, List[torch.Tensor]): Inputs of the model,
            can be single tensor or list of tensors, expected shape BxD or BxCxHxW.
        baseline (torch.Tensor): Baseline input, used to feed into model
            get output for comparison and backprogate gradients, expected shape same as inputs.
        target (orch.Tensor): Target label index for the batch,
            for classification task, this is the class id, expected shape one hot tensor BxD.
    Return:
        gradients (torch.Tensor, List[torch.Tensor])
    """
    xs = inputs
    bs = baseline
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

    pred = model(*xs)
    base_out = model(*bs)
    assert target.shape == pred.shape, "target shape not equal to output"
    # diff = pred * target - baseline
    output_dim = len(pred.shape) - 1
    diff = torch.sum(pred * target - base_out, -1 * output_dim)
    diff.backward()

    if torch.is_tensor(inputs):
        return xs[0].grad
    res = []
    for x in xs:
        res.append(x.grad)
    return res
