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

from hynet.layers.brew_layer import BrewLayer

from hynet.imgr.encoders.cifar10_wideresnet_encoder import Encoder
# from hynet.imgr.encoders.cifar10_resnet_encoder import Encoder
# from hynet.imgr.encoders.cifar10_cnn_encoder import Encoder
# from hynet.imgr.encoders.mnist_encoder import Encoder

class HynetImgrModel(AbsESPnetModel):
    """Image recognition model"""

    def __init__(self):
        assert check_argument_types()
        super().__init__()
        # task related
        self.brew_excute = False
        if self.brew_excute:
            self.max_iter = 2
        else:
            self.max_iter = 1
        # data related
        self.in_ch_sz = 32  # 28
        self.in_ch = 3  # 1
        self.out_ch = 10
        # network archictecture 
        self.encoder = Encoder()
        self.bias = True 
        self.decoder = BrewLayer(
            sample_size = self.encoder.img_size[0] * 
                          self.encoder.img_size[1] * 
                          self.encoder.out_channels,
            hidden_size = 256,
            target_size = self.out_ch,
            bias = self.bias)
        # cirterion fo task
        self.criterion = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def minimaxn(
        self, 
        x: torch.Tensor,
        dim: int
    ):
        max_x = torch.max(x, dim=dim, keepdim=True)[0]
        min_x = torch.min(x, dim=dim, keepdim=True)[0]

        norm = (max_x - min_x)
        norm[norm == 0.0] = 1.0

        x = (x - min_x) / norm
        
        return x

    def inv(
        self,
        x: torch.Tensor,
        label_hat: torch.Tensor,
        w_hat: torch.Tensor,
        split_to_np: bool
    ):
        assert len(label_hat.size()) == 3

        sign = torch.sign(x).unsqueeze(-1)
        w_hat *= sign

        if split_to_np:
            w_pos = torch.relu(w_hat)
            attn_pos = torch.matmul(label_hat, w_pos.transpose(-2, -1))
            attn_pos = attn_pos.squeeze(-2)

            w_neg = torch.relu(-1.0 * w_hat)
            attn_neg = torch.matmul(label_hat, w_neg.transpose(-2, -1))
            attn_neg = attn_neg.squeeze(-2)

            return attn_pos, attn_neg
        else:
            attn = torch.matmul(label_hat, w_hat.transpose(-2, -1))
            attn = attn.squeeze(-2)

            return attn

    def forward(
        self,
        image: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logger = {'imgs':[], 
                  'attns': [[],[],[]], 
                  'accs':[],
                  'accs_cs': 0.0, 
                  'loss_brew': 0.0}

        losses = []
        for i in range(self.max_iter):
            # 1. feedforward neural network
            feat, ratio = self.encoder(image)
            print(feat.size())
            exit()
            if i > 0:
                feat = feat * (1 - attn_pos.detach())
            b_sz, _, w, h = feat.size()
            feat_flat = feat.flatten(start_dim=1)
            label_hat, ratio = self.decoder(feat_flat)

            if self.brew_excute:
                # 2. brewing and check loss of p_hat results and normal result
                p_hat = self.decoder.brew(ratio=ratio)
                # 2.1 check brewing error
                label_hat_hat = torch.matmul(feat_flat.unsqueeze(-2), p_hat[0])
                label_hat_hat = label_hat_hat.squeeze(-2)
                if self.bias:
                    label_hat_hat += p_hat[1]
                logger['loss_brew'] = self.mse(label_hat, label_hat_hat).detach()
            
            # 3. caculate measurment 
            losses.append(self.criterion(label_hat, label))
            # 3.1 other measurment
            acc = self._calc_acc(label_hat, label)        
            logger['accs'].append(acc)

            # 4. inverse attention with feature
            if self.brew_excute:
                label_hat = label_hat - p_hat[1]
                label_hat = torch.softmax(label_hat, dim=-1).unsqueeze(-2)
                
                attn_pos = self.inv(feat_flat, label_hat, p_hat[0], split_to_np=False)

                attn_pos = torch.relu(attn_pos)
                attn_pos = attn_pos.view(b_sz, -1, w, h)
                attn_pos = attn_pos.flatten(start_dim=1)
                attn_pos = self.minimaxn(attn_pos, dim=1)
                attn_pos = attn_pos.view(b_sz, -1, w, h)
                    
                # 5. for logging
                if not self.training:
                    logger['imgs'].append(image.permute(0, 2, 3, 1))
                    logger['attns'][0].append(attn_pos.sum(1))
                    logger['attns'][1].append(feat.sum(1))
            else:
                if not self.training:
                    logger['imgs'].append(image.permute(0, 2, 3, 1))
                    logger['attns'][0].append(image.permute(0, 2, 3, 1))
                    logger['attns'][1].append(feat.sum(1))

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
