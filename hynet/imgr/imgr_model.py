from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import math

import torch
from torch import nn
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

from hynet.imgr.encoders.cifar10_vgg_encoder import EnDecoder

def minimaxn(x, dim):
    max_x = torch.max(x, dim=dim, keepdim=True)[0]
    min_x = torch.min(x, dim=dim, keepdim=True)[0]

    norm = (max_x - min_x)
    norm[norm == 0.0] = 1.0

    x = (x - min_x) / norm
    
    return x

class HynetImgrModel(AbsESPnetModel):
    """Image recognition model"""

    def __init__(self):
        assert check_argument_types()
        super().__init__()
        # task related
        self.brew_excute = True
        self.max_iter = 1
        # data related
        self.in_ch = 3  # 1
        self.out_ch = 10
        # network archictecture 
        self.model = EnDecoder(in_channels=self.in_ch,
                               num_classes=self.out_ch)
        # cirterion fo task
        self.criterion = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def shap(self, attn):
        attn_pos = torch.relu(attn)
        attn_neg = torch.relu(-1.0 * attn)
        
        # attn_pos = attn_pos.flatten(start_dim=2)
        # attn_pos = minimaxn(attn_pos, dim=2)
        # attn_pos = attn_pos.view(b_sz, -1, in_w, in_h)

        return attn_pos, attn_neg


    def forward(
        self,
        image: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logger = {'imgs':[], 
                  'attns': [[],[]], 
                  'accs':[],
                  'accs_cs': 0.0, 
                  'loss_brew': 0.0}

        losses = []
        for i in range(self.max_iter):
            b_sz, _, in_w, in_h = image.size()
            # 1. feedforward neural network
            logit, ratio, ratio_split_idx = self.model(image)
            if self.brew_excute:
                logit_hat = self.model.forward_linear(image, ratio.copy())
                logger['loss_brew'] = self.mse(logit, logit_hat).detach()
            
            # 3. caculate measurment 
            losses.append(self.criterion(logit, label))
            # 3.1 other measurment
            acc = self._calc_acc(logit, label)        
            logger['accs'].append(acc)

            # 4. inverse attention with feature
            if self.brew_excute:
                # TODO(j-pong): These line for non-bias free case. But we needs more smart way!
                # b_hat = self.model.brew_bias(self.model.decoder, ratio[ratio_split_idx:].copy())
                # logit_softmax = torch.softmax(logit - b_hat, dim=-1)
                
                logit_softmax = torch.softmax(logit, dim=-1)
                attn = self.model.backward_linear(logit_softmax, 
                                                  self.model.decoder, 
                                                  ratio[ratio_split_idx:])
                attn = attn.view(b_sz,
                                 self.model.out_channels,
                                 self.model.img_size[0],
                                 self.model.img_size[1])
                attn = self.model.backward_linear(attn, 
                                                  self.model.encoder, 
                                                  ratio[:ratio_split_idx])

                attn_pos, attn_neg = self.shap(attn)
                    
                # 5. for logging
                if not self.training:
                    image_ = image.permute(0, 2, 3, 1)
                    logger['imgs'].append(image_.sum(-1))
                    attn_pos = attn_pos.permute(0, 2, 3, 1)
                    logger['attns'][0].append(attn_pos.sum(-1))
                    attn_neg = attn_neg.permute(0, 2, 3, 1)
                    logger['attns'][1].append(attn_neg.sum(-1))
            else:
                if not self.training:
                    image_ = image.permute(0, 2, 3, 1)
                    logger['imgs'].append(image_)
                    logger['attns'][0].append(image[:,:,:,0])
                    logger['attns'][0].append(image[:,:,:,1])

        loss = 0.0
        for los in losses:
            loss += los

        stats = dict(
                loss=loss.detach(),
                loss_brew=logger['loss_brew'],
                acc_start=logger['accs'][0],
                acc_end=logger['accs'][-1],
                acc_cs=logger['accs_cs'] 
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
