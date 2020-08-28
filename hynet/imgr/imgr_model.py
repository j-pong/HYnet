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
# from hynet.imgr.masked_loss import MultiMaskLoss

class HynetImgrModel(AbsESPnetModel):
    """Image recognition model"""

    def __init__(self):
        assert check_argument_types()
        super().__init__()

        self.brew_layer = BrewLayer(
            sample_size=28*28,
            hidden_size=512,
            target_size=10)

        # self.criterion = MultiMaskLoss(
        #     criterion=nn.CrossEntropyLoss(reduction='none'))
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _calc_acc(self,
        y_hat: torch.Tensor, 
        y: torch.Tensor
    ):
        pred = y_hat.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(y.view_as(pred)).float().mean().item()
        return correct

    def forward(
        self,
        image: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = image.shape[0]
        image = image.float().view(batch_size, -1)

        label_hat = self.brew_layer(image)

        loss = self.criterion(label_hat, label)
        acc = self._calc_acc(label_hat, label)

        stats = dict(
            loss=loss.detach(),
            acc=acc
        )
        loss, stats, weight = force_gatherable(
            (loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {"feats": image}
