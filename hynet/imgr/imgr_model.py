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

from hynet.layers.brew_layer import BrewLayer, BrewCnnLayer, BrewAttLayer

class HynetImgrModel(AbsESPnetModel):
    """Image recognition model"""

    def __init__(self):
        assert check_argument_types()
        super().__init__()

        self.max_iter = 2

        self.cnn_hidden_sizes = [32, 64]
        self.cnn_kernel_sizes = [3, 3]
        self.wh = 24

        self.proj_hidden_size = 128
        self.bias = True

        self.brew_cnn_layer = BrewCnnLayer(
            kernel_size=self.cnn_kernel_sizes[0],
            hidden_size=self.cnn_hidden_sizes[0],
            in_channel=1,
            wh=26)
        self.brew_cnn_layer2 = BrewCnnLayer(
            kernel_size=self.cnn_kernel_sizes[1],
            hidden_size=self.cnn_hidden_sizes[1], 
            in_channel=self.cnn_hidden_sizes[0],
            wh=self.wh)

        self.flatten = nn.Flatten()

        self.brew_layer = BrewLayer(
            sample_size=self.wh * self.wh * self.cnn_hidden_sizes[1],
            hidden_size=self.proj_hidden_size,
            target_size=10,
            bias=self.bias)

        self.csim = nn.CosineSimilarity(dim=2)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion_mse = nn.MSELoss()

    def minimaxn(
        self, 
        x: torch.Tensor,
        dim: int
    ):
        max_x = torch.max(x, dim=dim, keepdim=True)[0]
        min_x = torch.min(x, dim=dim, keepdim=True)[0]
        x = (x - min_x) / (max_x - min_x)

        return x

    def inv(
        self,
        x,
        label_hat: torch.Tensor,
        w_hat: torch.Tensor
    ):
        assert len(label_hat.size()) == 3

        sign = torch.sign(x).unsqueeze(-1)
        sign[sign==0.0] = 1.0
        w_hat *= sign

        w_pos = torch.relu(w_hat)
        attn_pos = torch.matmul(label_hat, w_pos.transpose(-2, -1))
        attn_pos = attn_pos.squeeze(-2)

        w_neg = torch.relu(-1.0 * w_hat)
        attn_neg = torch.matmul(label_hat, w_neg.transpose(-2, -1))
        attn_neg = attn_neg.squeeze(-2)

        return attn_pos, attn_neg

    def forward(
        self,
        image: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = image.shape[0]
        image = image.view(batch_size, -1).float()

        logger = {'imgs':[], 
                  'attns': [[],[],[]], 
                  'accs':[],
                  'accs_cs': None, 
                  'loss_brew': None}

        losses = []
        for i in range(self.max_iter):
            # 0. prepare data
            if i == 0:
                image = self.minimaxn(image, dim=1).detach()
                image = image.view(batch_size, 1, 28, 28)

            # 1. feedforward neural network
            feat, _ = self.brew_cnn_layer(image)
            feat, _ = self.brew_cnn_layer2(feat)
            if i > 0:
                attn_pos = attn_pos.flatten(start_dim=2)
                attn_pos = self.minimaxn(attn_pos, dim=2)
                attn_pos = attn_pos.view(batch_size, -1, self.wh, self.wh)
                feat = feat * (1 - attn_pos.detach())
            feat_flat = self.flatten(feat)
            label_hat, ratio = self.brew_layer(feat_flat)

            # 2. brewing and check loss of p_hat results and normal result
            p_hat = self.brew_layer.brew(ratio=ratio)
            
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
            inverse_application = 'sup'
            if inverse_application == 'sup':
                label_hat = label_hat - p_hat[1]
                label_hat = torch.softmax(label_hat, dim=-1).unsqueeze(-2)
                
                attn_pos, attn_neg = self.inv(feat_flat, label_hat, p_hat[0])

                attn_pos = attn_pos.view(batch_size, self.cnn_hidden_sizes[1], self.wh, self.wh)
                attn_neg = attn_neg.view(batch_size, self.cnn_hidden_sizes[1], self.wh, self.wh)

                logger['accs_cs'] = 0.0

            elif inverse_application == 'sim':
                label_hat = F.one_hot(torch.arange(start=0, end=10), 
                                      num_classes=10).float().to(label.device)
                label_hat = label_hat.view(1, 10, 10)
                label_hat = label_hat.repeat(batch_size, 1, 1)

                attn_pos, attn_neg = self.inv(feat_flat, label_hat, p_hat[0])
                label_hat_hat_cs = self.csim(attn_pos, feat.flatten(start_dim=1).unsqueeze(1))

                logger['accs_cs'] = self._calc_acc(label_hat_hat_cs, label)     

                
            # 5. for logging
            if not self.training:
                if inverse_application == 'sim':
                    attn_pos = attn_pos[:, 0]
                    attn_neg = attn_neg[:, 0]

                logger['attns'][0].append(attn_pos[0].mean(-3))
                logger['attns'][0].append(attn_pos[0, 0])
                logger['attns'][0].append(attn_pos[0, 1])

                logger['attns'][1].append(attn_neg[0].mean(-3))
                logger['attns'][1].append(attn_neg[0, 0])
                logger['attns'][1].append(attn_neg[0, 1])

                logger['imgs'].append(feat[0].mean(-3))
                logger['imgs'].append(feat[0, 0])
                logger['imgs'].append(feat[0, 1])

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
