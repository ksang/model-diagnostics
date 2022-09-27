import torch
from torch.autograd import Variable
from typing import Union, List, Tuple


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
