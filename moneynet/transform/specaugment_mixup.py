"""Spec Augment module for preprocessing i.e., data augmentation"""

import random

import numpy
import torch
from moneynet.nets.pytorch_backend.ICT.nets_utils import mixup_data

def Mix_then_Spec(xs, hlens, ys_pad, mask, specmix_params, train=True):
    if not train:
        return xs, hlens, ys_pad, ys_pad, 1, mask

    mixup_alpha = specmix_params["mixup_alpha"]
    scheme = specmix_params["mixup_scheme"]
    F = specmix_params["F"]
    T = specmix_params["T"]

    xs, ys_pad, ys_pad_b, _, lam = mixup_data(xs, ys_pad, hlens, mixup_alpha, scheme)

    for idx in range(xs.shape[0]):
        cloned = xs[idx]

        # Time warp
        from PIL import Image
        from PIL.Image import BICUBIC
        device = cloned.device
        cloned = cloned.data.cpu().numpy()
        window = 5
        t = cloned.shape[0]
        if t - window > window:
            # NOTE: randrange(a, b) emits a, a + 1, ..., b - 1
            center = random.randrange(window, t - window)
            warped = random.randrange(center - window, center + window) + 1  # 1 ... t - 1

            left = Image.fromarray(cloned[:center]).resize((cloned.shape[1], warped), BICUBIC)
            right = Image.fromarray(cloned[center:]).resize((cloned.shape[1], t - warped), BICUBIC)

            cloned[:warped] = left
            cloned[warped:] = right
        cloned = torch.Tensor(cloned).to(device)

        # F mask
        n_mask = specmix_params["F_n_mask"]

        num_mel_channels = cloned.shape[1]
        fs = numpy.random.randint(0, F, size=(n_mask, 2))

        for f, mask_end in fs:
            f_zero = random.randrange(0, num_mel_channels - f)
            mask_end += f_zero

            # avoids randrange error if values are equal and range is empty
            if f_zero == f_zero + f:
                continue

            cloned[:, f_zero:mask_end] = 0

        # T mask
        n_mask = specmix_params["T_n_mask"]

        len_spectro = cloned.shape[0]
        ts = numpy.random.randint(0, T, size=(n_mask, 2))
        for t, mask_end in ts:
            # avoid randint range error
            if len_spectro - t <= 0:
                continue
            t_zero = random.randrange(0, len_spectro - t)

            # avoids randrange error if values are equal and range is empty
            if t_zero == t_zero + t:
                continue

            mask_end += t_zero
            cloned[t_zero:mask_end] = 0

        xs[idx] = cloned

    return xs, hlens, ys_pad, ys_pad_b, lam, mask

def Spec_then_Mix(xs, hlens, ys_pad, mask, specmix_params, train=True):
    if not train:
        return xs, hlens, ys_pad, ys_pad, 1, mask

    import numpy as np
    import torch

    mixup_alpha = specmix_params["mixup_alpha"]

    # TODO: add local scheme
    scheme = specmix_params["mixup_scheme"]
    F = specmix_params["F"]
    T = specmix_params["T"]

    xs_device = xs.device
    ys_pad_device = ys_pad.device

    if mixup_alpha > 0.:
        lam = np.random.uniform(mixup_alpha, 1)
    else:
        lam = 1.

    batch_size = xs.size()[0]

    xs = xs.data.cpu().numpy()
    ys_pad = ys_pad.data.cpu().numpy()

    index = torch.randperm(batch_size)
    xs_b = xs[index, :].copy()
    ys_pad_b = ys_pad[index, :].copy()

    for idx in range(xs.shape[0]):
        cloned = xs[idx]
        cloned_b = xs_b[idx]
        ys_cloned = ys_pad[idx]
        ys_cloned_b = ys_pad_b[idx]

        # F mask
        n_mask = specmix_params["F_n_mask"]


        num_mel_channels = cloned.shape[1]
        fs = numpy.random.randint(0, F, size=(n_mask, 2))

        for f, mask_end in fs:
            f_zero = random.randrange(0, num_mel_channels - f)
            mask_end += f_zero

            # avoids randrange error if values are equal and range is empty
            if f_zero == f_zero + f:
                continue

            cloned[:, f_zero:mask_end] = 0

        # T mask
        n_mask = specmix_params["T_n_mask"]

        len_spectro = cloned.shape[0]
        ts = numpy.random.randint(0, T, size=(n_mask, 2))
        for t, mask_end in ts:
            # avoid randint range error
            if len_spectro - t <= 0:
                continue
            t_zero = random.randrange(0, len_spectro - t)

            # avoids randrange error if values are equal and range is empty
            if t_zero == t_zero + t:
                continue

            mask_end += t_zero
            cloned[t_zero:mask_end] = lam * cloned[t_zero:mask_end] \
                + (1-lam) * cloned_b[t_zero:mask_end]

            ys_cloned[t_zero:mask_end] = ys_cloned_b[t_zero:mask_end]

        xs[idx] = cloned
        ys_pad_b[idx] = ys_cloned

    xs = torch.Tensor(xs).to(xs_device)
    ys_pad_b = torch.Tensor(ys_pad_b).to(ys_pad_device).long()
    ys_pad = torch.Tensor(ys_pad).to(ys_pad_device).long()

    return xs, hlens, ys_pad, ys_pad_b, lam, mask

def Mixup(xs, hlens, ys_pad, mask, specmix_params, train=True):
    if not train:
        return xs, hlens, ys_pad, ys_pad, 1, mask

    mixup_alpha = specmix_params["mixup_alpha"]
    scheme = specmix_params["mixup_scheme"]

    xs, ys_pad, ys_pad_b, _, lam = mixup_data(xs, ys_pad, hlens, mixup_alpha, scheme)

    return xs, hlens, ys_pad, ys_pad_b, lam, mask

def Spec(xs, hlens, ys_pad, mask, specmix_params, train=True):
    if not train:
        return xs, hlens, ys_pad, ys_pad, 1, mask

    F = specmix_params["F"]
    T = specmix_params["T"]

    for idx in range(xs.shape[0]):
        cloned = xs[idx]

        # Time warp
        from PIL import Image
        from PIL.Image import BICUBIC
        device = cloned.device
        cloned = cloned.data.cpu().numpy()
        window = 5
        t = cloned.shape[0]
        if t - window > window:
            # NOTE: randrange(a, b) emits a, a + 1, ..., b - 1
            center = random.randrange(window, t - window)
            warped = random.randrange(center - window, center + window) + 1  # 1 ... t - 1

            left = Image.fromarray(cloned[:center]).resize((cloned.shape[1], warped), BICUBIC)
            right = Image.fromarray(cloned[center:]).resize((cloned.shape[1], t - warped), BICUBIC)

            cloned[:warped] = left
            cloned[warped:] = right
        cloned = torch.Tensor(cloned).to(device)

        # F mask
        n_mask = specmix_params["F_n_mask"]

        num_mel_channels = cloned.shape[1]
        fs = numpy.random.randint(0, F, size=(n_mask, 2))

        for f, mask_end in fs:
            f_zero = random.randrange(0, num_mel_channels - f)
            mask_end += f_zero

            # avoids randrange error if values are equal and range is empty
            if f_zero == f_zero + f:
                continue

            cloned[:, f_zero:mask_end] = cloned.mean()

        # T mask
        n_mask = specmix_params["T_n_mask"]

        len_spectro = cloned.shape[0]
        ts = numpy.random.randint(0, T, size=(n_mask, 2))
        for t, mask_end in ts:
            # avoid randint range error
            if len_spectro - t <= 0:
                continue
            t_zero = random.randrange(0, len_spectro - t)

            # avoids randrange error if values are equal and range is empty
            if t_zero == t_zero + t:
                continue

            mask_end += t_zero
            cloned[t_zero:mask_end] = cloned.mean()

        xs[idx] = cloned

    return xs, hlens, ys_pad, ys_pad, 1, mask

def Spec_Mix_Sep(xs, hlens, ys_pad, mask, specmix_params, train=True):
    if not train:
        return xs, hlens, ys_pad, ys_pad, 1, mask
    augment = numpy.random.randint(0, 2)
    if augment == 0:
        xs, hlens, ys_pad, ys_pad_b, lam, mask = Mixup(xs, hlens, ys_pad, mask, specmix_params, train=train)
    elif augment == 1:
        xs, hlens, ys_pad, ys_pad_b, lam, mask = Spec(xs, hlens, ys_pad, mask, specmix_params, train=train)
    return xs, hlens, ys_pad, ys_pad_b, lam, mask

def Specmix(xs_pad, ilens, ys_pad, src_mask, specmix_params, train=True):
    args = (xs_pad, ilens, ys_pad, src_mask, specmix_params, train)
    if specmix_params["augmentation"] == "specmix":
        xs_pad, ilens, ys_pad, ys_pad_b, lam, src_mask = Spec_then_Mix(*args)
    elif specmix_params["augmentation"] == "mixspec":
        xs_pad, ilens, ys_pad, ys_pad_b, lam, src_mask = Mix_then_Spec(*args)
    elif specmix_params["augmentation"] == "mixup":
        xs_pad, ilens, ys_pad, ys_pad_b, lam, src_mask = Mixup(*args)
    elif specmix_params["augmentation"] == "specaug":
        xs_pad, ilens, ys_pad, ys_pad_b, lam, src_mask = Mixup(*args)
    elif specmix_params["augmentation"] == "separate":
        xs_pad, ilens, ys_pad, ys_pad_b, lam, src_mask = Spec_Mix_Sep(*args)

    return xs_pad, ilens, ys_pad, ys_pad_b, lam, src_mask
