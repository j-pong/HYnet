from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

from hynet.imgr.brew_layer import BrewLayer, BrewCnnLayer

class HynetImgrModel(AbsESPnetModel):
    """Image recognition model"""

    def __init__(self):
        assert check_argument_types()
        super().__init__()

        self.bias = True
        self.max_iter = 1

        self.brew_cnn_layer = BrewCnnLayer()
        self.brew_layer = BrewLayer(
            sample_size=18*18*10,
            hidden_size=512,
            target_size=10,
            bias=self.bias)
        # self.brew_recon_layer = BrewLayer(
        #     sample_size=18*18*10,
        #     hidden_size=512,
        #     target_size=28*28,
        #     bias=self.bias)    

        self.criterion = nn.CrossEntropyLoss()
        self.criterion_mse = nn.MSELoss()

    def minimaxn(
        self, 
        x: torch.Tensor
    ):
        max_x = torch.max(x, dim=-1, keepdim=True)[0].detach()
        min_x = torch.min(x, dim=-1, keepdim=True)[0].detach()
        x = (x - min_x) / (max_x - min_x)

        return x

    def forward(
        self,
        image: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = image.shape[0]
        image = image.view(batch_size, -1).float()

        imgs = [[],[]]
        attns = [[],[]]
        accs = []

        losses = []
        for i in range(self.max_iter):
            image = self.minimaxn(image).detach()

            # 1. feedforward neural network 
            feat, ratio_cnn = self.brew_cnn_layer(image)
            feat = torch.flatten(feat, start_dim=1)
            label_hat, ratio = self.brew_layer(feat)

            # 2. brewing and check loss of p_hat results and normal result
            p_hat = self.brew_layer.brew(ratio=ratio)
            # p_hat = self.brew_cnn_layer.brew(ratio=ratio_cnn, w_hat=p_hat[0], b_hat=p_hat[1])
            if i == 0:
                label_hat_hat = torch.matmul(feat.unsqueeze(-2), p_hat[0])
                label_hat_hat = label_hat_hat.squeeze(-2)
                if self.bias:
                    label_hat_hat += p_hat[1]
                loss_brew = torch.pow(label_hat - label_hat_hat, 2).mean() 

            # 3. caculate measurment 
                loss = self.criterion(label_hat, label)
                losses.append(loss)
            acc = self._calc_acc(label_hat, label)        
            accs.append(acc)

            # 4. inverse atte ntion with feature
            label_hat = label_hat - p_hat[1]
            label_hat = torch.softmax(label_hat, dim=-1)
            w_pos = torch.relu(p_hat[0])
            attn_pos = torch.matmul(label_hat.unsqueeze(-2), w_pos.transpose(-2, -1))
            attn_pos = attn_pos.squeeze(-2)
            w_neg = torch.relu(-1.0 * p_hat[0])
            attn_neg = torch.matmul(label_hat.unsqueeze(-2), w_neg.transpose(-2, -1))
            attn_neg = attn_neg.squeeze(-2)
            attns[0].append(attn_pos[0].view(18, 18, 10).mean(-1))
            attns[1].append(attn_neg[0].view(18, 18, 10).mean(-1))

            # if i == 0:
            #     attn = [attn_pos, attn_neg]
            #     image_recon, ratio = self.brew_recon_layer(torch.cat(attn, dim=-1).detach())
            #     image_recon = image_recon + attn_pos.detach() - attn_neg.detach()
            #     loss_recong = self.criterion_mse(image_recon, image)
            #     losses.append(loss_recong)

            # image = attn_pos * image
            imgs[0].append(image[0].view(28, 28))
            # imgs[1].append(image_recon[0].view(28, 28))
            imgs[1].append(feat[0].view(18, 18, 10).mean(-1))

        loss = 0.0
        for los in losses:
            loss += los

        stats = dict(
                loss=loss.detach(),
                loss_brew=loss_brew.detach(),
                acc_start=accs[0],
                acc_end=accs[-1]
            )
        if not self.training:
            stats['aux'] = [imgs[1], attns[0], attns[1]]

        loss, stats, weight = force_gatherable(
            (loss, stats, batch_size), loss.device)
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
