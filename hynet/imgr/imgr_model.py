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
from hynet.imgr.masked_loss import MultiMaskLoss


class HynetImgrModel(AbsESPnetModel):
    """Image recognition model"""

    def __init__(self):
        assert check_argument_types()
        super().__init__()

        self.brew_layer = BrewLayer(
            sample_size=28*28,
            hidden_size=512,
            target_size=10)

        self.creiterion = MultiMaskLoss(
            criterion=nn.CrossEntropyLoss(reduction='none'))

    def forward(
        self,
        img: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        batch_size = img.shape[0]

        label_hat = self.brew_layer(img)

        loss = self.critertion(label, label_hat)

        stats = dict(
            loss=loss.detach()
        )
        loss, stats, weight = force_gatherable(
            (loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        img: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {"feats": img}
