from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import math
import copy

import torch
from torch import nn
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

from hynet.imgr.encoders.cifar10_vgg_encoder import EnDecoder
# from hynet.imgr.encoders.mnist_vgg_encoder import EnDecoder
from hynet.layers.brew import attn_norm

from captum.attr import IntegratedGradients, LayerIntegratedGradients, NeuronIntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift, DeepLiftShap
from captum.attr import NoiseTunnel


def attribute_image_features(net, target, algorithm, input, **kwargs):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=target,
                                              **kwargs
                                             )
    
    return tensor_attributions
class HynetImgrModel(AbsESPnetModel):
    """Image recognition model"""

    def __init__(self, 
                 xai_excute, 
                 xai_mode, 
                 cfg_type, 
                 bias):
        assert check_argument_types()
        super().__init__()
        # task related
        self.xai_excute = xai_excute
        if self.xai_excute:
            self.max_iter = 3
        else:
            self.max_iter = 1
        self.xai_mode = xai_mode
        self.cfg_type = cfg_type
        self.bias = bias

        # data related
        self.in_ch = 3
        self.out_ch = 10

        # network archictecture 
        self.model = EnDecoder(in_channels=self.in_ch,
                               num_classes=self.out_ch,
                               bias=self.bias,
                               model_type=self.cfg_type)
                               
        # cirterion fo task
        self.criterion = nn.CrossEntropyLoss()

    def pn_decomp(self, attn):
        attn_pos = torch.relu(attn)
        attn_neg = torch.relu(-1.0 * attn)

        return attn_pos, attn_neg

    def forward(
        self,
        image: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logger = {'imgs':[], 
                  'attns': [[],[]], 
                  'accs':[]}

        losses = []
        b_sz, in_ch, in_h, in_w = image.size()
        assert self.in_ch == in_ch

        for i in range(self.max_iter):
            if self.xai_excute:
                # 1. preprocessing with each iteration
                if i > 0:
                    max_img = torch.max(image, dim=1, keepdim=True)[0]
                    min_img = torch.min(image, dim=1, keepdim=True)[0]
                    attn = attn_norm(attn).detach()
                    image = attn_norm(image * attn, max_img, min_img)
                # 2. forward pass for training network
                if self.xai_mode == 'lrp':
                    logit, ratios = self.model.forward_lrp(image)
                elif self.xai_mode == 'brew':
                    if self.bias:
                        if i > 0:
                            bm = True
                        else:
                            bm = False
                    else:
                        bm = False
                    logit = self.model.forward(image, bias_mask=bm, mode='brew')
                else:
                    logit = self.model.forward(image)
            else:
                logit = self.model.forward(image)
            
            # 3. caculate measurment 
            if i == 0: 
                losses.append(self.criterion(logit, label))
            acc = self._calc_acc(logit, label)        
            logger['accs'].append(acc)

            # inverse attention with feature 
            if self.xai_excute:
                with torch.no_grad():
                    if self.xai_mode == 'brew':
                        # label generation
                        logit_softmax = torch.softmax(logit, dim=-1)
                        mask = F.one_hot(label, num_classes=self.out_ch).bool()
                        
                        logit_softmax_cent = logit_softmax.masked_fill(~mask, 0.0)
                        attn_cent = self.model.backward_linear(image, logit_softmax_cent)

                        logit_softmax_other = logit_softmax.masked_fill(mask, 0.0)
                        attn_other = self.model.backward_linear(image, logit_softmax_other)
                        
                        attn = attn_cent
                    elif self.xai_mode == 'lrp':
                        # label generation
                        logit_softmax = torch.softmax(logit, dim=-1)
                        mask = F.one_hot(label, num_classes=self.out_ch).bool()

                        logit_softmax_cent = logit_softmax.masked_fill(~mask, 0.0)
                        attn_cent = self.backward_lrp(logit_softmax_cent, copy.deepcopy(ratios))
                        logit_softmax_other = logit_softmax.masked_fill(mask, 0.0)
                        attn_other = self.backward_lrp(logit_softmax_other, ratios)

                        attn = attn_cent
                    elif self.xai_mode == 'saliency':
                        saliency = Saliency(self.model)
                        grads = saliency.attribute(image, target=label)
                        attn = grads.squeeze()
                    elif self.xai_mode == 'ig':
                        ig = IntegratedGradients(self.model)
                        attr_ig, delta = attribute_image_features(self.model, label, ig, image, 
                                                                  baselines=image * 0, 
                                                                  return_convergence_delta=True)
                        attn = attr_ig.squeeze()
                    elif self.xai_mode == 'ig_nt':
                        ig = IntegratedGradients(self.model)
                        nt = NoiseTunnel(ig)
                        attr_ig_nt = attribute_image_features(self.model, label, nt, image, 
                                                                baselines=image * 0, 
                                                                nt_type='smoothgrad',
                                                                n_samples=4, stdevs=0.02)
                        attn = attr_ig_nt.squeeze(0)
                    elif self.xai_mode == 'dl':
                        dl = DeepLift(self.model)
                        attr_dl = attribute_image_features(self.model, label, dl, image, 
                                                            baselines=image * 0)
                        attn = attr_dl.squeeze(0)
                    elif self.xai_mode == 'dls':
                        dl = DeepLiftShap(self.model)
                        attr_dl, delta = attribute_image_features(self.model, label, dl, image, 
                                                            baselines=image * 0, 
                                                            return_convergence_delta=True)
                        attn = attr_dl.squeeze(0)
                    
                    attn_pos, attn_neg = self.pn_decomp(attn)
                        
                    # 5. for logging
                    if not self.training:
                        image_ = image.permute(0, 2, 3, 1)
                        logger['imgs'].append(image_.sum(-1))
                        attn1 = attn.permute(0, 2, 3, 1)
                        logger['attns'][0].append(attn1.sum(-1))
                        attn2 = attn_neg.permute(0, 2, 3, 1)
                        logger['attns'][1].append(attn2.sum(-1))

            else: 
                if not self.training:
                    image_ = image.permute(0, 2, 3, 1)
                    logger['imgs'].append(image_)
                    logger['attns'][0].append(image_[:,:,:,0])
                    logger['attns'][1].append(image_[:,:,:,1])

        loss = 0.0
        for los in losses:
            loss += los

        if self.max_iter > 1:
            stats = dict(
                    loss=loss.detach(),
                    acc_start=logger['accs'][0],
                    acc_mid=logger['accs'][1],
                    acc_end=logger['accs'][-1]
                )
        else:
            stats = dict(
                    loss=loss.detach(),
                    acc_start=logger['accs'][0]
                )
                
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
        pred = y_hat.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
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

        attn_pos, attn_neg = self.inv(feat_flat, label_hat, p_hat[0], split_to_np=False)
        label_hat_hat_cs = F.cosine_similarity(attn_pos, feat_flat.unsqueeze(1), dim=2)

        return self._calc_acc(label_hat_hat_cs, label)

    def collect_feats(
        self,
        image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {"feats": image}
