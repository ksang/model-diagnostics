import torch
import torch.nn as nn
from typing import Union, List
from diagnostics.utils import initialize_inputs_baseline


def get_input_gradients(
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
        baseline (torch.Tensor, List[torch.Tensor]): Baseline input, used to feed into model
            get output for comparison and backprogate gradients, expected shape same as inputs.
        target (torch.Tensor): Target label index for the batch,
            for classification task, this is the class id, expected shape one hot tensor BxD.
    Return:
        gradients (torch.Tensor, List[torch.Tensor])
    """
    xs, bs = initialize_inputs_baseline(inputs, baseline)

    pred = model(*xs)
    base_out = model(*bs)
    assert target.shape == pred.shape, "target shape not equal to output"
    batch_size = pred.shape[0]
    diff = torch.sum((pred * target - base_out).view(batch_size, -1), -1)
    model.zero_grad()
    diff.sum().backward()

    if torch.is_tensor(inputs):
        return xs[0].grad
    return [x.grad for x in xs]
