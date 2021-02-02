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

from hynet.imgr.models.vgg import EnDecoder as Vgg
from hynet.imgr.models.resnet import EnDecoder as Resnet
from hynet.imgr.models.simnet import EnDecoder as Simnet

from captum.attr import IntegratedGradients
from captum.attr import Saliency, GuidedBackprop
from captum.attr import DeepLift, DeepLiftShap
from captum.attr import NoiseTunnel

from hynet.attr.bg import BrewGradient, GradientxInput
from hynet.imgr.imgr_model import HynetImgrModel as pt_model

class HynetImgrModel(AbsESPnetModel):
    """Image recognition model"""

    def __init__(self,
                 xai_excute,
                 xai_mode,
                 xai_iter,
                 st_excute,
                 cfg_type,
                 batch_norm,
                 bias,
                 in_ch,
                 out_ch):
        assert check_argument_types()
        super().__init__()
        # task related
        self.max_iter = xai_iter
        self.xai_mode = xai_mode
        self.st_excute = st_excute
        self.cfg_type = cfg_type

        self.batch_norm = batch_norm
        self.bias = bias

        # data related
        self.in_ch = in_ch
        self.out_ch = out_ch

        # network archictecture
        if self.cfg_type == 'wrn50_2' or self.cfg_type == 'wrn40_4' or self.cfg_type == 'wrn28_10':
            self.model = Resnet(in_channels=self.in_ch,
                                num_classes=self.out_ch,
                                batch_norm=self.batch_norm,
                                bias=self.bias,
                                model_type=self.cfg_type)
        elif self.cfg_type == 'simnet':
            self.model = Simnet(in_channels=self.in_ch,
                             num_classes=self.out_ch,
                             batch_norm=self.batch_norm,
                             bias=self.bias,
                             model_type=self.cfg_type)
        else:
            self.model = Vgg(in_channels=self.in_ch,
                             num_classes=self.out_ch,
                             batch_norm=self.batch_norm,
                             bias=self.bias,
                             model_type=self.cfg_type)

        self.pt_model = pt_model(
            0, xai_mode, xai_iter,
            0, 'B2', batch_norm, bias, in_ch, out_ch
            )
        ckpt = torch.load(
            '/home/Workspace/HYnet/egs/xai/cifar10/exp/imgr_train_vgg7_bnwob/checkpoint.pth', 
            map_location=f"cuda:{torch.cuda.current_device()}"
            )
        self.pt_model.load_state_dict(ckpt['model'])
        self.pt_model = self.pt_model.model
        self.pt_model.train(False)
        for param in self.pt_model.parameters():
            param.requires_grad = False

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
        focused_layer = model.focused_layer
        attn_hook_handle = None

        mask_prod = torch.ones_like(image)

        for i in range(self.max_iter):
            model.zero_grad()

            def attn_apply(self, x):
                return x[0] * mask_prod
            attn_hook_handle = focused_layer.register_forward_pre_hook(
                attn_apply)

            logit = model(image)
            acc = self._calc_acc(logit, label)
            logger['accs'].append(acc)

            if self.xai_mode == 'saliency':
                if attn_hook_handle is not None:
                    attn_hook_handle.remove()
                    attn_hook_handle = None
                image_ = image * mask_prod
                saliency = Saliency(model)
                grads = saliency.attribute(image_, target=label)
                attn = grads.squeeze()
                attn = image * attn
                loss_brew = 0.0
            elif self.xai_mode == 'bg':
                if attn_hook_handle is not None:
                    attn_hook_handle.remove()
                    attn_hook_handle = None
                image_ = image * mask_prod
                bg = BrewGradient(model)
                attr_bg = bg.attribute(image_,
                                       layer=focused_layer,
                                       target=label)

                attn = attr_bg.squeeze()
                loss_brew = bg.loss_brew
            elif self.xai_mode == 'gxi':
                if attn_hook_handle is not None:
                    attn_hook_handle.remove()
                    attn_hook_handle = None
                image_ = image * mask_prod
                gxi = GradientxInput(model)
                attr_gxi = gxi.attribute(image_,
                                       layer=focused_layer,
                                       target=label)

                attn = attr_gxi.squeeze()
                loss_brew = gxi.loss_brew
            elif self.xai_mode == 'gbp':
                if attn_hook_handle is not None:
                    attn_hook_handle.remove()
                    attn_hook_handle = None
                image_ = image * mask_prod
                gbp = GuidedBackprop(model)
                attr_bg = gbp.attribute(image_,
                                        target=label)
                attn = attr_bg.squeeze()
                loss_brew = 0.0

            # recasting to float type
            attn = attn.float()

            mask = self.feat_minmax_norm(attn, 1.0, 0.0)
            image_ = (image * mask_prod).permute(0, 2, 3, 1)
            mask_prod *= mask

            logger['loss_brew'].append(loss_brew)

            if not self.training:
                logger['imgs'].append(image_.sum(-1))
                attn1 = attn.permute(0, 2, 3, 1)
                logger['attns'][0].append(attn1.sum(-1))
                attn2 = mask.permute(0, 2, 3, 1)
                logger['attns'][1].append(attn2.sum(-1))

        return logger, mask_prod

    def forward(
        self,
        image: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.pt_model.train(False)

        logger = {
                'loss_brew': [],
                'imgs': [],
                'attns': [[], []],
                'accs': []
            }

        b_sz, in_ch, in_h, in_w = image.size()
        assert self.in_ch == in_ch

        # 1. feedforward
        logger, attns = self.forward_xai(self.pt_model, image, label, logger)
        logit = self.model(image * attns.detach().clone())

        # 2. caculate measurment
        loss = self.criterion(logit, label)
        acc = self._calc_acc(logit, label)

        stats = dict(
                loss=loss.detach(),
                acc=acc,
            )
        for i in range(0, self.max_iter):
            stats['acc_iter{}'.format(i)] = logger['accs'][i]
            stats['loss_brew{}'.format(i)] = logger['loss_brew'][i]
        if not self.training:
            stats['aux'] = [logger['imgs'],
                            logger['attns'][0],
                            logger['attns'][1]]

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

    def _calc_sim_acc(
        self,
        feat_flat,
        label,
        p_hat
    ):
        label_hat = F.one_hot(torch.arange(start=0, end=self.out_ch),
                              num_classes=self.out_ch).to(label.device)
        label_hat = label_hat.float()
        label_hat = label_hat.view(1, self.out_ch, self.out_ch)
        label_hat = label_hat.repeat(feat_flat.size(0), 1, 1)

        attn_pos, attn_neg = self.inv(
            feat_flat, label_hat, p_hat[0], split_to_np=False)
        label_hat_hat_cs = F.cosine_similarity(
            attn_pos, feat_flat.unsqueeze(1), dim=2)

        return self._calc_acc(label_hat_hat_cs, label)

    def collect_feats(
        self,
        image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {"feats": image}
