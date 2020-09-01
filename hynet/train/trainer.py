import argparse
import dataclasses
from dataclasses import is_dataclass
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.train.reporter import Reporter
from espnet2.train.reporter import SubReporter

from espnet2.train.trainer import Trainer, ReduceOp, TrainerOptions


class ImgrTrainer(Trainer):
    @classmethod
    @torch.no_grad()
    def validate_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: TrainerOptions,
    ) -> None:
        assert check_argument_types()
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = isinstance(model, torch.nn.parallel.DistributedDataParallel)

        model.eval()

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        for (_, batch) in iterator:
            assert isinstance(batch, dict), type(batch)
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            _, stats, weight = model(**batch)
            del stats['aux']
            if ngpu > 1 or distributed:
                # Apply weighted averaging for stats.
                # if distributed, this method can also apply all_reduce()
                stats, weight = recursive_average(stats, weight, distributed)

            reporter.register(stats, weight)
            reporter.next()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)

        import matplotlib.pyplot as plt

        ind_plot = 0
        for ids, batch in iterator:
            _, stats, _ = model(**batch)
            img_list = stats['aux']

            plt.clf()
            len_max = len(img_list[0])
            for i, img in enumerate(img_list[0]):
                plt.subplot(3, len_max, i+1)
                plt.imshow(img.detach().cpu().numpy())
                plt.colorbar()
            for j, img in enumerate(img_list[1]):
                plt.subplot(3, len_max, i+j+2)
                plt.imshow(img.detach().cpu().numpy())
                plt.colorbar()
            for k, img in enumerate(img_list[2]):
                plt.subplot(3, len_max, i+j+k+3)
                plt.imshow(img.detach().cpu().numpy())
                plt.colorbar()
            ind_plot += 1
            plt.savefig('imgs{}.png'.format(ind_plot))

            if ind_plot == 3:
                break
