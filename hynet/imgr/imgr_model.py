from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import nn
from typeguard import check_argument_types

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

from hynet.imgr.brew_layer import BrewLayer

class HynetImgrModel(AbsESPnetModel):
    """Image recognition model"""

    def __init__(self):
        assert check_argument_types()
        super().__init__()

        self.brew_layer = BrewLayer(
            sample_size=28*28,
            hidden_size=512,
            target_size=10,
            bias=False)

        self.brew_layer2 = BrewLayer(
            sample_size=28*28,
            hidden_size=512,
            target_size=10,
            bias=False)

        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        image: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = image.shape[0]
        image = image.view(batch_size, -1).float()
        image = self.minimaxn(image).detach()

        # 1. feedforward neural network 
        label_hat, ratio = self.brew_layer(image)

        # 2. brewing and check loss of p_hat results and normal result
        p_hat = self.brew_layer.brew(ratio=ratio)
        label_hat_hat = torch.matmul(image.unsqueeze(-2), p_hat[0])
        label_hat_hat = label_hat_hat.squeeze(-2)
        loss_brew = torch.pow(label_hat - label_hat_hat, 2).mean()  

        # 3. caculate measurment 
        loss = self.criterion(label_hat, label)
        acc = self._calc_acc(label_hat, label)

        # 4. inverse atte ntion with feature
        attn = torch.matmul(torch.softmax(label_hat, dim=-1).unsqueeze(-2), p_hat[0].abs().transpose(-2, -1))
        attn = attn.squeeze(-2)
        context = attn * image
        context = self.minimaxn(context).detach()

        # 5. re-feedforward neural network 
        label_hat, ratio = self.brew_layer2(context)
        p_hat = self.brew_layer2.brew(ratio=ratio)

        # 6. caculate measurment 
        loss2 = self.criterion(label_hat, label)
        acc2 = self._calc_acc(label_hat, label)

        # 7. inverse atte ntion with feature
        attn2 = torch.matmul(torch.softmax(label_hat, dim=-1).unsqueeze(-2), p_hat[0].abs().transpose(-2, -1))
        attn2 = attn2.squeeze(-2)

        import matplotlib.pyplot as plt
        plt.clf()
        plt.subplot(221)
        plt.imshow(attn[0].view(28,28).detach().cpu().numpy(), aspect='auto')
        plt.colorbar()
        plt.subplot(222)
        plt.imshow(image[0].view(28,28).detach().cpu().numpy(), aspect='auto')
        plt.colorbar()
        plt.subplot(223)
        plt.imshow(context[0].view(28,28).detach().cpu().numpy(), aspect='auto')
        plt.colorbar()
        plt.subplot(224)
        plt.imshow(attn2[0].view(28,28).detach().cpu().numpy(), aspect='auto')
        plt.colorbar()
        plt.savefig("test.png")

        stats = dict(
            loss=loss.detach(),
            acc=acc,
            loss2=loss2.detach(),
            acc2=acc2,
            loss_brew=loss_brew.detach(),
        )

        loss += loss2
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
    
    def minimaxn(self, x):
        max_x = torch.max(x, dim=-1, keepdim=True)[0].detach()
        min_x = torch.min(x, dim=-1, keepdim=True)[0].detach()
        # if (max_x - min_x) == 0.0:
        #     logging.warning('Divided by the zero with max-min value : {}, Thus return None'.format((max_x - min_x)))
        #     x = None
        # else:
        x = (x - min_x) / (max_x - min_x)

        return x
