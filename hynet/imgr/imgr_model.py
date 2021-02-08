from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import warnings

from collections import OrderedDict

import math
import copy

import torch
from torch import nn
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

from hynet.imgr.models.nib_vgg import EnDecoder as NibVgg
from hynet.imgr.models.vgg import EnDecoder as Vgg
from hynet.imgr.models.wrn import EnDecoder as WideResNet

from captum.attr import IntegratedGradients
from captum.attr import Saliency, GuidedBackprop
from captum.attr import DeepLift, DeepLiftShap
from captum.attr import NoiseTunnel

from hynet.attr.custom_gba import GradientxSingofInput, GradientxInput


class HynetImgrModel(AbsESPnetModel):
    """Image recognition model"""

    def __init__(self,
                 xai_excute: int,
                 xai_mode: str,
                 xai_iter: int,
                 st_excute: int,
                 cfg_type: str,
                 batch_norm: int,
                 bias: int,
                 in_ch: int,
                 out_ch: int,
                 ):
        assert check_argument_types()
        super().__init__()
        # task related
        self.xai_excute = xai_excute
        if self.xai_excute:
            self.max_iter = xai_iter
        else:
            self.max_iter = 1
        self.xai_mode = xai_mode
        self.st_excute = st_excute
        self.cfg_type = cfg_type

        self.batch_norm = batch_norm
        self.bias = bias

        # data related
        self.in_ch = in_ch
        self.out_ch = out_ch

        # network archictecture
        if self.cfg_type.find('nib') != -1:
            self.model = NibVgg(in_channels=self.in_ch,
                                num_classes=self.out_ch,
                                batch_norm=self.batch_norm,
                                bias=self.bias,
                                model_type=self.cfg_type.split('_')[-1])
        else:
            self.model = Vgg(in_channels=self.in_ch,
                             num_classes=self.out_ch,
                             batch_norm=self.batch_norm,
                             bias=self.bias,
                             model_type=self.cfg_type)

        # cirterion fo task
        self.criterion = nn.CrossEntropyLoss()

    def feat_minmax_norm(self, x, max_y=1.0, min_y=0.0):
        # save original shape of feature
        b_sz, ch, in_h, in_w = x.size()

        # flatten for whole space dimension
        x = x.flatten(start_dim=2)

        # get min-max value
        max_x = torch.max(x, dim=2, keepdim=True)[0]
        min_x = torch.min(x, dim=2, keepdim=True)[0]

        # normalization element
        norm = (max_x - min_x)
        denorm = (max_y - min_y)
        norm[norm == 0.0] = 1.0

        # normalization
        x = (x - min_x) / norm * denorm + min_y
        x = x.view(b_sz, ch, in_h, in_w)

        return x

    @torch.no_grad()
    def forward_xai(
        self,
        model,
        image: torch.Tensor,
        label: torch.Tensor,
        logger
    ):
        mask_prod = torch.ones_like(image)

        for i in range(self.max_iter):
            model.zero_grad()

            image_ = image * mask_prod

            if self.xai_mode == 'pgxi':
                saliency = Saliency(model)

                grads = saliency.attribute(image_, target=label)

                attn = grads.squeeze()
                attn = image * attn
                
                delta = 0.0
            elif self.xai_mode == 'ig':
                ig = IntegratedGradients(model)

                attr_ig, delta = ig.attribute(image_,
                                              baselines=image * 0,
                                              target=label,
                                              return_convergence_delta=True)
                
                attn = attr_ig.squeeze()

                delta = 0.0
            elif self.xai_mode == 'dl':
                dl = DeepLift(model)
                
                attr_dl, delta = dl.attribute(image_,
                                              baselines=image * 0,
                                              target=label,
                                              return_convergence_delta=True)

                attn = attr_dl.squeeze(0)
                
                delta = 0.0
            elif self.xai_mode == 'gxsi':
                bg = GradientxSingofInput(model)

                attr_bg = bg.attribute(image_,
                                       target=label)

                attn = attr_bg.squeeze()

                delta = bg.loss_brew
            elif self.xai_mode == 'gxi':                
                gxi = GradientxInput(model)

                attr_gxi = gxi.attribute(image_,
                                         target=label)

                attn = attr_gxi.squeeze()

                delta = gxi.loss_brew
            elif self.xai_mode == 'gbp':
                gbp = GuidedBackprop(model)
                
                attr_bg = gbp.attribute(image_,
                                        target=label)

                attn = attr_bg.squeeze()

                delta = 0.0
            else:
                raise AttributeError("This attribution method is not supported!")
            
            # Measure the MIA
            logit = model(image_)
            acc = self._calc_acc(logit, label)

            # Normalzation the attributions for obtaining the attrbituion mask
            attn = attn.float()
            mask = self.feat_minmax_norm(attn, 1.0, 0.0)

            # Update the attribution mask
            mask_prod *= mask
            
            logger['accs'].append(acc)

            if not self.training:
                image_ = image_.permute(0, 2, 3, 1)
                logger['imgs'].append(image_.sum(-1))

        model.zero_grad()

        return logger

    def forward(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        return_plot: bool=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        logger = {
            'accs': [],
            'imgs': []
        }

        b_sz, in_ch, in_h, in_w = image.size()
        assert self.in_ch == in_ch

        # 1. Feedforward
        logit = self.model(image)

        # 2. Coculate loss and accuracy
        loss = self.criterion(logit, label)
        acc = self._calc_acc(logit, label)

        stats = dict(
                loss_iter0=loss.detach(),
                acc_iter0=acc
                )

        # 3. Attributions are caculated by the GBA methods
        if not self.training and self.xai_excute:
            assert self.model.training == False

            logger = self.forward_xai(self.model, image, label, logger)

            for i in range(1, self.max_iter):
                stats['acc_iter{}'.format(i)] = logger['accs'][i]

            if return_plot:
                stats['imgs'] = logger['imgs']

        loss, stats, weight = force_gatherable(
            (loss, stats, b_sz), loss.device)
        return loss, stats, weight

    def _calc_acc(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor
    ):
        # get the index of the max log-probability
        pred = y_hat.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).float().mean().item()
        return correct

    def collect_feats(
        self,
        image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {"feats": image}
