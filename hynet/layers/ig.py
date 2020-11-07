#!/usr/bin/env python3
import typing
from typing import Any, Callable, Tuple, Union

from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn.functional as F

from captum.attr._utils.common import _run_forward, _format_input, _select_targets
from captum.attr._utils.gradient import apply_gradient_requirements, undo_gradient_requirements

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
        layer, 
        target: TargetType = None,
        return_convergence_delta: bool = True,
        precision: str = 'float32'
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:  
        # precision casting for accurate calculation
        if precision == 'float32':
            inputs = inputs.float()
            self.forward_func.float()
        elif precision == 'float64':
            inputs = inputs.double()
            self.forward_func.double()
        else:
            raise AttributeError()
        
        # gradient setting 
        gv = {}
        inputs = _format_input(inputs)
        gradient_mask = apply_gradient_requirements(inputs)

        # augmented gradient hooks
        def save_val_hook(self, x, y):
            gv['val'] = x[0]
        def save_grad_hook(self, x, y):
            gv['grad'] = x[0]

        #save grad to specific layer
        layer.register_forward_hook(save_val_hook)
        layer.register_backward_hook(save_grad_hook)
        
        with torch.autograd.set_grad_enabled(True):
            # runs forward pass
            outputs = _run_forward(self.forward_func, inputs, target)
            assert outputs[0].numel() == 1, (
                "Target not provided when necessary, cannot"
                " take gradient with respect to multiple outputs."
            )
            # calculate gradient
            torch.autograd.grad(torch.unbind(outputs), inputs)

        # back to the original precision
        if precision == 'float32':
            pass
        elif precision == 'float64':
            self.forward_func.float()
        else:
            raise AttributeError()

        # initialization hooks
        layer._forward_hooks = OrderedDict()
        layer._backward_hooks = OrderedDict()

        # clear gradient and detach from graph
        undo_gradient_requirements(inputs, gradient_mask)

        # Efficiency propertiy check
        if return_convergence_delta:
            outputs_hat = (gv['val'] * gv['grad']).flatten(start_dim=1).sum(-1, keepdim=True)
            loss_brew = F.mse_loss(outputs, outputs_hat)
            self.loss_brew = loss_brew

        # importance of feature has sign field 
        sign = torch.sign(gv['val'])
        attn = gv['grad'] * sign

        # flush dictionary
        gv = {}
        return attn

