#!/usr/bin/env python3
import typing
from typing import Any, Callable, Tuple, Union

from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn.functional as F

from captum._utils.common import _run_forward, _format_input, _select_targets
from captum._utils.gradient import apply_gradient_requirements, undo_gradient_requirements

from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.attribution import GradientAttribution
from captum.attr._utils.batching import _batched_operator

from captum._utils.typing import (
    BaselineType,
    Literal,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

class GradientxInput(GradientAttribution):
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
        # gv = {}
        inputs = _format_input(inputs)
        gradient_mask = apply_gradient_requirements(inputs)
        
        with torch.autograd.set_grad_enabled(True):
            # runs forward pass
            outputs = _run_forward(self.forward_func, inputs, target)
            assert outputs[0].numel() == 1, (
                "Target not provided when necessary, cannot"
                " take gradient with respect to multiple outputs."
            )
            # calculate gradient
            grads = torch.autograd.grad(torch.unbind(outputs), inputs)[0]

        # clear gradient and detach from graph
        undo_gradient_requirements(inputs, gradient_mask)

        # back to the original precision
        if precision == 'float32':
            pass
        elif precision == 'float64':
            self.forward_func.float()
        else:
            raise AttributeError()
        
        # output type check
        if isinstance(inputs, tuple):
            inputs = inputs[0]

        if return_convergence_delta:
            outputs_hat = (inputs * grads).flatten(start_dim=1).sum(-1, keepdim=True)
            loss_brew = F.mse_loss(outputs, outputs_hat)
            self.loss_brew = loss_brew

        # importance of feature has sign field 
        grads = grads * inputs

        return grads

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
        inputs = _format_input(inputs)
        gradient_mask = apply_gradient_requirements(inputs)
        
        with torch.autograd.set_grad_enabled(True):
            # runs forward pass
            outputs = _run_forward(self.forward_func, inputs, target)
            assert outputs[0].numel() == 1, (
                "Target not provided when necessary, cannot"
                " take gradient with respect to multiple outputs."
            )
            # calculate gradient
            grads = torch.autograd.grad(torch.unbind(outputs), inputs)[0]

        # clear gradient and detach from graph
        undo_gradient_requirements(inputs, gradient_mask)

        # back to the original precision
        if precision == 'float32':
            pass
        elif precision == 'float64':
            self.forward_func.float()
        else:
            raise AttributeError()
        
        # output type check
        if isinstance(inputs, tuple):
            inputs = inputs[0]

        if return_convergence_delta:
            outputs_hat = (inputs * grads).flatten(start_dim=1).sum(-1, keepdim=True)
            loss_brew = F.mse_loss(outputs, outputs_hat)
            self.loss_brew = loss_brew

        # importance of feature has sign field 
        sign = torch.sign(inputs)
        grads = grads * sign

        return grads