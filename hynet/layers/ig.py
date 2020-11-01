#!/usr/bin/env python3
import typing
from typing import Any, Callable, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F

from captum.attr._utils.common import _run_forward, _format_input

from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.attribution import GradientAttribution
from captum.attr._utils.batching import _batched_operator
from captum.attr._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_attributions,
    _format_input_baseline,
    _is_tuple,
    _reshape_and_sum,
    _validate_input,
)
from captum.attr._utils.typing import (
    BaselineType,
    Literal,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

class BrewGradient(GradientAttribution):
    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:

            forward_func (callable):  The forward function of the model or any
                          modification of it
        """
        GradientAttribution.__init__(self, forward_func)
        self.loss_brew = 0.0
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        return_convergence_delta: bool = False,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:  
        with torch.autograd.set_grad_enabled(True):
            inputs = inputs.requires_grad_()
            # runs forward pass
            outputs = _run_forward(self.forward_func, inputs, target)
            assert outputs[0].numel() == 1, (
                "Target not provided when necessary, cannot"
                " take gradient with respect to multiple outputs."
            )
            grads = torch.autograd.grad(torch.unbind(outputs), inputs)
        grads = grads[0].detach()
        outputs_hat = (inputs * grads).flatten(start_dim=1).sum(-1, keepdim=True)
        self.loss_brew = F.mse_loss(outputs_hat, outputs).detach()

        sign = torch.sign(inputs)
        grads = grads * sign

        return grads

