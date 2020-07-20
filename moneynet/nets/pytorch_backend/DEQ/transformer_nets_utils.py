from collections import defaultdict
import numpy as np

import torch
from torch import nn
from torch.autograd import Function

from moneynet.nets.pytorch_backend.DEQ.broyden import broyden, analyze_broyden


class RootFind(Function):
    """ Generic DEQ module that uses Broyden's method to find the equilibrium state """

    @staticmethod
    def f(func, xs_pad, masks):
        return func(xs_pad, masks)

    @staticmethod
    def g(func, xs_pad, masks):
        return RootFind.f(func, xs_pad, masks) - xs_pad

    @staticmethod
    def broyden_find_root(func, xs_pad, masks, eps, *args):
        batch, seq_len, cdim = xs_pad.size()
        z1ss_est = xs_pad.clone().detach()
        threshold = args[-2]  # Can also set this to be different, based on training/inference
        train_step = args[-1]

        g = lambda x: RootFind.g(func, x, masks)
        result_info = broyden(g, z1ss_est, threshold=threshold, eps=eps, name="forward")
        z1ss_est = result_info['result']

        if threshold > 100:
            torch.cuda.empty_cache()
        return z1ss_est.clone().detach()

    @staticmethod
    def forward(ctx, func, xs_pad, masks, *args):
        batch, seq_len, cdim = xs_pad.size()
        eps = 1e-6 * np.sqrt(batch * seq_len * cdim)
        root_find = RootFind.broyden_find_root
        ctx.args_len = len(args)
        with torch.no_grad():
            z1ss_est = root_find(func, xs_pad, masks, eps, *args)  # args include pos_emb, threshold, train_step

            # If one would like to analyze the convergence process (e.g., failures, stability), should
            # insert here or in broyden_find_root.
            return z1ss_est

    @staticmethod
    def backward(ctx, grad_z1):
        grad_args = [None for _ in range(ctx.args_len)]
        return (None, grad_z1, None, None, *grad_args)


class DEQModule(nn.Module):
    """
    The equilibrium solver module. Forward pass is unspecified; we provide an implementation of the
    implicit differentiation through the equilibrium point in the inner `Backward` class.
    """

    def __init__(self, func, func_copy):
        super(DEQModule, self).__init__()
        self.func = func
        self.func_copy = func_copy

    def forward(self, z1s, masks, **kwargs):
        raise NotImplemented

    class Backward(Function):
        """
        A 'dummy' function that does nothing in the forward pass and perform implicit differentiation
        in the backward pass. Essentially a wrapper that provides backprop for the `DEQModule` class.
        You should use this inner class in DEQModule's forward() function by calling:

            self.Backward.apply(self.func_copy, ...)

        """

        @staticmethod
        def forward(ctx, func_copy, z1ss, masks, *args):
            ctx.save_for_backward(z1ss, masks)
            ctx.func = func_copy
            ctx.args = args
            return z1ss

        @staticmethod
        def backward(ctx, grad):
            torch.cuda.empty_cache()

            # grad should have dimension (bsz x d_model x seq_len)
            bsz, seq_len, d_model = grad.size()
            grad = grad.clone()
            z1ss, masks = ctx.saved_tensors
            args = ctx.args
            threshold, train_step = args[-2:]

            func = ctx.func
            z1ss = z1ss.clone().detach().requires_grad_()

            with torch.enable_grad():
                y = RootFind.g(func, z1ss, masks)

            def g(x):
                y.backward(x, retain_graph=True)  # Retain for future calls to g
                JTx = z1ss.grad.clone().detach()
                z1ss.grad.zero_()
                return JTx + grad

            eps = 2e-10 * np.sqrt(bsz * seq_len * d_model)
            dl_df_est = torch.zeros_like(grad)

            result_info = broyden(g, dl_df_est, threshold=threshold, eps=eps, name="backward")
            dl_df_est = result_info['result']

            y.backward(torch.zeros_like(dl_df_est), retain_graph=False)

            grad_args = [None for _ in range(len(args))]
            return (None, dl_df_est, None, None, *grad_args)

