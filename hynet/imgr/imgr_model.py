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

from hynet.layers.brew_layer import BrewLayer, BrewCnnLayer, BrewAttLayer
from hynet.imgr.encoders.cifar10_vgg_encoder import Encoder
# from hynet.imgr.encoders.mnist_encoder import Encoder

class HynetImgrModel(AbsESPnetModel):
    """Image recognition model"""

    def __init__(self):
        assert check_argument_types()
        super().__init__()
        # task related
        self.max_iter = 2
        # data related
        self.in_ch_sz = 32  # 28
        self.in_ch = 3  # 1
        self.out_ch = 10
        # network archictecture 
        self.encoder = Encoder()
        self.bias = True 
        self.decoder = BrewLayer(
            sample_size = self.encoder.img_size[0] * self.encoder.img_size[1] * 256,
            hidden_size = 256,
            target_size = self.out_ch,
            bias = self.bias)
        # cirterion fo task
        self.criterion = nn.CrossEntropyLoss()

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
        w_hat: torch.Tensor
    ):
        assert len(label_hat.size()) == 3

        sign = torch.sign(x).unsqueeze(-1)
        # sign[sign == 0.0] = 1.0
        w_hat *= sign

        # w_pos = torch.relu(w_hat)
        # attn_pos = torch.matmul(label_hat, w_pos.transpose(-2, -1))
        # attn_pos = attn_pos.squeeze(-2)

        # w_neg = torch.relu(-1.0 * w_hat)
        # attn_neg = torch.matmul(label_hat, w_neg.transpose(-2, -1))
        # attn_neg = attn_neg.squeeze(-2)

        # return attn_pos, attn_neg

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
                  'accs_cs': None, 
                  'loss_brew': None}

        losses = []
        for i in range(self.max_iter):
            # 0. prepare data
            if i == 0:
                with torch.no_grad():
                    image = image.float()
                    image = image.flatten(start_dim=2)
                    image = self.minimaxn(image, dim=2)           
                    image = image.view(image.size(0), self.in_ch, self.in_ch_sz, self.in_ch_sz)

            # 1. feedforward neural network
            feat = self.encoder(image)
            if i > 0:
                feat = feat * attn_pos.detach()
            feat_flat = feat.flatten(start_dim=1)
            label_hat, ratio = self.decoder(feat_flat)

            # 2. brewing and check loss of p_hat results and normal result
            p_hat = self.decoder.brew(ratio=ratio)
            
            # 2.1 check brewing error
            if i == 0:
                label_hat_hat = torch.matmul(feat_flat.unsqueeze(-2), p_hat[0])
                label_hat_hat = label_hat_hat.squeeze(-2)
                if self.bias:
                    label_hat_hat += p_hat[1]
                logger['loss_brew'] = torch.pow(label_hat - label_hat_hat, 2).mean()
                
                # 3. caculate measurment 
                losses.append(self.criterion(label_hat, label))
            # 3.1 other measurment
            acc = self._calc_acc(label_hat, label)        
            logger['accs'].append(acc)

            # 4. inverse atte ntion with feature
            label_hat = label_hat - p_hat[1]
            label_hat = torch.softmax(label_hat, dim=-1).unsqueeze(-2)
            
            attn_pos = self.inv(feat_flat, label_hat, p_hat[0])

            b_sz, _, w, h = feat.size()
            attn_pos = attn_pos.view(b_sz, -1, w, h)
            # attn_neg = attn_neg.view(b_sz, -1, w, h)

            attn_pos = attn_pos.flatten(start_dim=1)
            # attn_pos = attn_pos - attn_neg.flatten(start_dim=2)
            attn_pos = self.minimaxn(attn_pos, dim=1)
            attn_pos = attn_pos.view(b_sz, -1, w, h)

            logger['accs_cs'] = 0.0

            # elif inverse_application == 'sim':
            #     label_hat = F.one_hot(torch.arange(start=0, end=self.out_ch), 
            #                           num_classes=self.out_ch).float().to(label.device)
            #     label_hat = label_hat.view(1, self.out_ch, self.out_ch)
            #     label_hat = label_hat.repeat(b_sz, 1, 1)

            #     attn_pos, attn_neg = self.inv(feat_flat, label_hat, p_hat[0])
            #     label_hat_hat_cs = F.cosine_similarity(attn_pos, feat.flatten(start_dim=1).unsqueeze(1), dim=2)

            #     logger['accs_cs'] = self._calc_acc(label_hat_hat_cs, label)     

                
            # 5. for logging
            if not self.training:
                # if inverse_application == 'sim':
                #     attn_pos = attn_pos[:, 0]
                #     attn_neg = attn_neg[:, 0]

                logger['attns'][0].append(attn_pos[0].mean(0))

                # logger['attns'][1].append(attn_neg[0].mean(0))
                logger['attns'][1].append(attn_pos[0].mean(0))

                logger['imgs'].append(image[0].transpose(1,0).transpose(2,1))
                # logger['imgs'].append(feat[0].mean(0))
                # logger['imgs'].append(feat[0, 0])
                # logger['imgs'].append(feat[0, 1])

        loss = 0.0
        for los in losses:
            loss += los

        stats = dict(
                loss=loss.detach(),
                loss_brew=logger['loss_brew'].detach(),
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

    def collect_feats(
        self,
        image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {"feats": image}
