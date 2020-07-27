#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Training/decoding definition for the speech recognition task."""

import copy
import json
import logging
import math
import os
import sys

from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions
from chainer.training.updater import StandardUpdater

import numpy as np

from tensorboardX import SummaryWriter

import torch
from torch.nn.parallel import data_parallel

from kaldi_io import write_mat

# from espnet.asr.asr_utils import adadelta_eps_decay
# from espnet.asr.asr_utils import add_results_to_json
# from espnet.asr.asr_utils import CompareValueTrigger
# from espnet.asr.asr_utils import format_mulenc_args
# from espnet.asr.asr_utils import get_model_conf
# from espnet.asr.asr_utils import plot_spectrogram
# from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot
# from espnet.asr.pytorch_backend.asr_init import load_trained_model
# from espnet.asr.pytorch_backend.asr_init import load_trained_modules
# import espnet.lm.pytorch_backend.extlm as extlm_pytorch
# from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.e2e_asr import pad_list
# import espnet.nets.pytorch_backend.lm.default as lm_pytorch
# from espnet.nets.pytorch_backend.streaming.segment import SegmentStreamingE2E
# from espnet.nets.pytorch_backend.streaming.window import WindowStreamingE2E
# from espnet.transform.spectrogram import IStft
# from espnet.transform.transformation import Transformation
# from espnet.utils.cli_writers import file_writer_helper
from espnet.utils.dataset import ChainerDataLoader
from espnet.utils.dataset import TransformDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.evaluator import BaseEvaluator
from espnet.utils.training.iterators import ShufflingEnabler
# from espnet.utils.training.tensorboard_logger import TensorboardLogger
# from espnet.utils.training.train_utils import check_early_stop
# from espnet.utils.training.train_utils import set_early_stop

from moneynet.utils.chainer_pipe.io_utils import LoadInputsAndTargets

import matplotlib

matplotlib.use("Agg")

if sys.version_info[0] == 2:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest as zip_longest


def _recursive_to(xs, device):
    if torch.is_tensor(xs):
        return xs.to(device)
    if isinstance(xs, tuple):
        return tuple(_recursive_to(x, device) for x in xs)
    return xs


class CustomEvaluator(BaseEvaluator):
    """Custom Evaluator for Pytorch.

    Args:
        model (torch.nn.Module): The model to evaluate.
        iterator (chainer.dataset.Iterator) : The train iterator.

        target (link | dict[str, link]) :Link object or a dictionary of
            links to evaluate. If this is just a link object, the link is
            registered by the name ``'main'``.

        device (torch.device): The device used.
        ngpu (int): The number of GPUs.

    """

    def __init__(self, model, iterator, target, device, ngpu=None):
        super(CustomEvaluator, self).__init__(iterator, target)
        self.model = model
        self.device = device
        if ngpu is not None:
            self.ngpu = ngpu
        elif device.type == "cpu":
            self.ngpu = 0
        else:
            self.ngpu = 1

    # The core part of the update routine can be customized by overriding
    def evaluate(self):
        """Main evaluate routine for CustomEvaluator."""
        iterator = self._iterators["main"]

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, "reset"):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch in it:
                x = _recursive_to(batch, self.device)
                observation = {}
                with reporter_module.report_scope(observation):
                    # read scp files
                    # x: original json with loaded features
                    #    will be converted to chainer variable later
                    if self.ngpu == 0:
                        self.model(*x)
                    else:
                        # apex does not support torch.nn.DataParallel
                        data_parallel(self.model, x, range(self.ngpu))

                summary.add(observation)
        self.model.train()

        return summary.compute_mean()


class CustomUpdater(StandardUpdater):
    """Custom Updater for Pytorch.

    Args:
        model (torch.nn.Module): The model to update.
        grad_clip_threshold (float): The gradient clipping value to use.
        train_iter (chainer.dataset.Iterator): The training iterator.
        optimizer (torch.optim.optimizer): The training optimizer.

        device (torch.device): The device to use.
        ngpu (int): The number of gpus to use.
        use_apex (bool): The flag to use Apex in backprop.

    """

    def __init__(
            self,
            model,
            grad_clip_threshold,
            train_iter,
            optimizer,
            device,
            ngpu,
            grad_noise=False,
            accum_grad=1,
            use_apex=False,
    ):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.grad_clip_threshold = grad_clip_threshold
        self.device = device
        self.ngpu = ngpu
        self.accum_grad = accum_grad
        self.forward_count = 0
        self.grad_noise = grad_noise
        self.iteration = 0
        self.use_apex = use_apex

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Main update routine of the CustomUpdater."""
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")
        epoch = train_iter.epoch

        # Get the next batch (a list of json files)
        batch = train_iter.next()
        # self.iteration += 1 # Increase may result in early report,
        # which is done in other place automatically.
        x = _recursive_to(batch, self.device)
        is_new_epoch = train_iter.epoch != epoch
        # When the last minibatch in the current epoch is given,
        # gradient accumulation is turned off in order to evaluate the model
        # on the validation set in every epoch.
        # see details in https://github.com/espnet/espnet/pull/1388

        # Compute the loss at this time step and accumulate it
        if self.ngpu == 0:
            loss = self.model(*x).mean() / self.accum_grad
        else:
            # apex does not support torch.nn.DataParallel
            loss = (
                    data_parallel(self.model, x, range(self.ngpu)).mean() / self.accum_grad
            )
        if self.use_apex:
            from apex import amp

            # NOTE: for a compatibility with noam optimizer
            opt = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # gradient noise injection
        if self.grad_noise:
            from espnet.asr.asr_utils import add_gradient_noise

            add_gradient_noise(
                self.model, self.iteration, duration=100, eta=1.0, scale_factor=0.55
            )

        # update parameters
        self.forward_count += 1
        if not is_new_epoch and self.forward_count != self.accum_grad:
            return
        self.forward_count = 0
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_threshold
        )
        logging.info("grad norm={}".format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning("grad norm is nan. Do not update model.")
        else:
            optimizer.step()
        optimizer.zero_grad()

    def update(self):
        self.update_core()
        # #iterations with accum_grad > 1
        # Ref.: https://github.com/espnet/espnet/issues/777
        if self.forward_count == 0:
            self.iteration += 1


class CustomConverter(object):
    """Custom batch converter for Pytorch.

    Args:
        subsampling_factor (int): The subsampling factor.
        dtype (torch.dtype): Data type to convert.

    """

    def __init__(self, tnum=1, subsampling_factor=1, dtype=torch.float32):
        """Construct a CustomConverter object."""
        self.subsampling_factor = subsampling_factor
        self.ignore_id = -1
        self.dtype = dtype
        self.tnum = tnum

    def __call__(self, batch, device=torch.device("cpu")):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor)

        """
        # batch should be located in list
        assert len(batch) == 1
        xs, ys = batch[0]

        # perform subsampling
        if self.subsampling_factor > 1:
            xs = [x[:: self.subsampling_factor, :] for x in xs]

        if len(xs) == 2:
            logging.info("input and target are different form by transform")
            xs_in = xs[0]
            xs_out = xs[1]
        else:
            xs_in = xs
            xs_out = xs

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs_in])
        if self.tnum > 0:
            xs_pad_in = pad_list(
                [torch.from_numpy(x[:-self.tnum]).float()
                 for x in xs_in],
                0
            ).to(device, dtype=self.dtype)

            xs_pad_out = pad_list(
                [
                    torch.stack([torch.from_numpy(x[i + 1:-self.tnum + i + 1]).float()
                                 if (-self.tnum + i + 1) != 0 else torch.from_numpy(x[i + 1:]).float()
                                 for i in range(self.tnum)], dim=1)
                    for x in xs_out],
                0
            ).to(device, dtype=self.dtype)
        else:
            xs_pad_in = pad_list(
                [torch.from_numpy(x).float()
                 for x in xs_in],
                0
            ).to(device, dtype=self.dtype)
            xs_pad_out = pad_list(
                [torch.from_numpy(x).float().unsqueeze(1)
                 for x in xs_out],
                0
            ).to(device, dtype=self.dtype)

        ilens = torch.from_numpy(ilens).to(device) - self.tnum
        # NOTE: this is for multi-output (e.g., speech translation)
        ys_pad = pad_list(
            [
                torch.from_numpy(
                    np.array(y[0][:]) if isinstance(y, tuple) else y
                ).long()
                for y in ys
            ],
            self.ignore_id,
        ).to(device)

        return xs_pad_in, xs_pad_out, ilens, ys_pad


def train(args):
    """Train with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning("cuda is not available")

    # get input and output dimension info
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]["input"][0]["shape"][-1])
    odim = int(valid_json[utts[0]]["output"][0]["shape"][-1])
    logging.info("#output dims: " + str(odim))

    # dynamic model
    model_class = dynamic_import(args.model_module)
    model = model_class(idim, idim, args)

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + "/model.json"
    with open(model_conf, "wb") as f:
        logging.info("writing a model config file to " + model_conf)
        f.write(
            json.dumps(
                (idim, odim, vars(args)),
                indent=4,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf_8")
        )
    for key in sorted(vars(args).keys()):
        logging.info("ARGS: " + key + ": " + str(vars(args)[key]))

    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        if args.batch_size != 0:
            logging.warning(
                "batch size is automatically increased (%d -> %d)"
                % (args.batch_size, args.batch_size * args.ngpu)
            )
            args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    if args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, args.train_dtype)
    else:
        dtype = torch.float32
    model = model.to(device=device, dtype=dtype)

    # Setup an optimizer
    if args.opt == "adadelta":
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps, weight_decay=args.weight_decay
        )
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    elif args.opt == "noam":
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt

        optimizer = get_std_opt(
            model, args.hdim, args.warmup_steps, args.lr
        )
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    # setup apex.amp
    if args.train_dtype in ("O0", "O1", "O2", "O3"):
        try:
            from apex import amp
        except ImportError as e:
            logging.error(
                f"You need to install apex for --train-dtype {args.train_dtype}. "
                "See https://github.com/NVIDIA/apex#linux"
            )
            raise e
        if args.opt == "noam":
            model, optimizer.optimizer = amp.initialize(
                model, optimizer.optimizer, opt_level=args.train_dtype
            )
        else:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=args.train_dtype
            )
        use_apex = True
    else:
        use_apex = False

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # Setup a converter
    converter = CustomConverter(tnum=args.tnum, subsampling_factor=model.subsample[0], dtype=dtype)

    # read json data
    with open(args.train_json, "rb") as f:
        train_json = json.load(f)["utts"]
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # make minibatch list (variable length)
    train = make_batchset(
        train_json,
        args.batch_size,
        args.maxlen_in,
        args.maxlen_out,
        args.minibatches,
        min_batch_size=args.ngpu if args.ngpu > 1 else 1,
        shortest_first=use_sortagrad,
        count=args.batch_count,
        batch_bins=args.batch_bins,
        batch_frames_in=args.batch_frames_in,
        batch_frames_out=args.batch_frames_out,
        batch_frames_inout=args.batch_frames_inout,
        iaxis=0,
        oaxis=0,
    )
    valid = make_batchset(
        valid_json,
        args.batch_size,
        args.maxlen_in,
        args.maxlen_out,
        args.minibatches,
        min_batch_size=args.ngpu if args.ngpu > 1 else 1,
        count=args.batch_count,
        batch_bins=args.batch_bins,
        batch_frames_in=args.batch_frames_in,
        batch_frames_out=args.batch_frames_out,
        batch_frames_inout=args.batch_frames_inout,
        iaxis=0,
        oaxis=0,
    )

    load_tr = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=args.preprocess_conf,
        preprocess_args={"train": True},  # Switch the mode of preprocessing
    )
    load_cv = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=args.preprocess_conf,
        preprocess_args={"train": False},  # Switch the mode of preprocessing
    )
    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    # default collate function converts numpy array to pytorch tensor
    # we used an empty collate function instead which returns list
    train_iter = ChainerDataLoader(
        dataset=TransformDataset(train, lambda data: converter([load_tr(data)])),
        batch_size=1,
        num_workers=args.n_iter_processes,
        shuffle=not use_sortagrad,
        collate_fn=lambda x: x[0],
    )
    valid_iter = ChainerDataLoader(
        dataset=TransformDataset(valid, lambda data: converter([load_cv(data)])),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0],
        num_workers=args.n_iter_processes,
    )

    # Set up a trainer
    updater = CustomUpdater(
        model,
        args.grad_clip,
        {"main": train_iter},
        optimizer,
        device,
        args.ngpu,
        args.grad_noise,
        args.accum_grad,
        use_apex=use_apex,
    )
    trainer = training.Trainer(updater, (args.epochs, "epoch"), out=args.outdir)
    if use_sortagrad:
        trainer.extend(ShufflingEnabler([train_iter]),
                       trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, "epoch"), )

    # Resume from a snapshot
    if args.resume:
        logging.info("resumed from %s" % args.resume)
        torch_resume(args.resume, trainer)

    # Evaluate the model with the test dataset for each epoch
    if args.save_interval_iters > 0:
        trainer.extend(CustomEvaluator(model, {"main": valid_iter}, reporter, device, args.ngpu),
                       trigger=(args.save_interval_iters, "iteration"), )
    else:
        trainer.extend(CustomEvaluator(model, {"main": valid_iter}, reporter, device, args.ngpu))

    data = sorted(
        list(valid_json.items())[: args.num_save_attention],
        key=lambda x: int(x[1]["input"][0]["shape"][1]),
        reverse=True,
    )
    vis_fn = model.calculate_images
    plot_class = model.images_plot_class
    img_reporter = plot_class(
        vis_fn,
        data,
        args.outdir + "/images",
        converter=converter,
        transform=load_cv,
        device=device,
    )
    trainer.extend(img_reporter, trigger=(args.report_interval_iters, "iteration"))
    # trainer.extend(img_reporter, trigger=(1, "epoch"))

    trainer.extend(extensions.PlotReport(["main/loss",
                                          "validation/main/loss"], "epoch", file_name="loss.png", ))

    # Save best models
    trainer.extend(
        snapshot_object(model, "model.loss.best"),
        trigger=training.triggers.MinValueTrigger("validation/main/loss"),
    )

    # save snapshot which contains model and optimizer states
    if args.save_interval_iters > 0:
        trainer.extend(torch_snapshot(filename="snapshot.iter.{.updater.iteration}"),
                       trigger=(args.save_interval_iters, "iteration"), )
    else:
        trainer.extend(torch_snapshot(), trigger=(1, "epoch"))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(args.report_interval_iters, "iteration")))
    report_keys = [
        "epoch",
        "iteration",
        "main/loss_1",
        "validation/main/loss_1",
        "main/loss_2",
        "validation/main/loss_2",
        "main/loss_3",
        "validation/main/loss_3",
        "main/loss_4",
        "validation/main/loss_4",
        "elapsed_time",
    ]
    if args.opt == "adadelta":
        trainer.extend(
            extensions.observe_value("eps",
                                     lambda trainer: trainer.updater.get_optimizer("main").param_groups[0]["eps"], ),
            trigger=(args.report_interval_iters, "iteration"), )
        report_keys.append("eps")
    trainer.extend(extensions.PrintReport(report_keys), trigger=(args.report_interval_iters, "iteration"), )
    trainer.extend(extensions.ProgressBar(update_interval=args.report_interval_iters))

    # Run the training
    trainer.run()


def recog(args):
    set_deterministic_pytorch(args)

    from espnet.asr.asr_utils import get_model_conf, torch_load
    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # load trained model parameters
    model_class = dynamic_import(train_args.model_module)
    model = model_class(idim, odim, train_args)
    torch_load(args.model, model)
    # model, train_args = load_trained_model(args.model)
    model.recog_args = args

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info("gpu id: " + str(gpu_id))
        model.cuda()

    # read json data
    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=None,
        preprocess_args={"train": False},
    )

    ark_file = open(args.result_ark, 'wb')
    if args.batchsize == 0:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info("(%d/%d) decoding " + name, idx, len(js.keys()))
                batch = [(name, js[name])]

                feat = load_inputs_and_targets(batch)
                feat = (feat[0][0])

                hyps = model.recognize(feat)
                hyps = hyps.squeeze(0)
                hyps = hyps.data.numpy()

                write_mat(ark_file, hyps, key=name)
