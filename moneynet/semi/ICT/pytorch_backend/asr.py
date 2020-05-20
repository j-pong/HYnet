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
import codecs

from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions
from chainer.training.updater import StandardUpdater
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.nn.parallel import data_parallel
from kaldi_io import write_mat

from espnet.asr.asr_utils import adadelta_eps_decay
from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import format_mulenc_args
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import plot_spectrogram
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.asr.pytorch_backend.asr_init import load_trained_modules
import espnet.lm.pytorch_backend.extlm as extlm_pytorch
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.e2e_asr import pad_list
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.nets.pytorch_backend.streaming.segment import SegmentStreamingE2E
from espnet.nets.pytorch_backend.streaming.window import WindowStreamingE2E
from espnet.transform.spectrogram import IStft
from espnet.transform.transformation import Transformation
from espnet.utils.cli_writers import file_writer_helper
from espnet.utils.dataset import ChainerDataLoader
from espnet.utils.dataset import TransformDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.evaluator import BaseEvaluator
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

from moneynet.nets.pytorch_backend.ICT.nets_utils import update_ema_variables
from moneynet.nets.pytorch_backend.ICT.nets_utils import cosine_rampdown

import matplotlib

matplotlib.use("Agg")


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
        ul_iterator = self._iterators["sub"]

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, "reset"):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        if hasattr(ul_iterator, "reset"):
            ul_iterator.reset()
            ul_it = ul_iterator
        else:
            ul_it = copy.copy(ul_iterator)

        summary = reporter_module.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch, ul_batch in zip(it, ul_it):
                x = _recursive_to(batch, self.device)
                ul_x = _recursive_to(ul_batch, self.device)
                observation = {}
                with reporter_module.report_scope(observation):
                    # read scp files
                    # x: original json with loaded features
                    #    will be converted to chainer variable later
                    if self.ngpu == 0:
                        self.model(*x, *ul_x, None)
                    else:
                        # apex does not support torch.nn.DataParallel
                        data_parallel(self.model, (*x, *ul_x, None), range(self.ngpu))

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
    def __init__(self, model, grad_clip_threshold, train_iter,
                 optimizer, device, ngpu, ICT_args, grad_noise=False, accum_grad=1, use_apex=False,
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

        # Semi-supervised related
        self.consistency_rampup_ends = ICT_args["consistency_rampup_ends"]
        self.cosine_rampdown_starts = ICT_args["cosine_rampdown_starts"]
        self.cosine_rampdown_ends = ICT_args["cosine_rampdown_ends"]
        self.ema_pre_decay = ICT_args["ema_pre_decay"]
        self.ema_post_decay = ICT_args["ema_post_decay"]

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Main update routine of the CustomUpdater."""
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch (a list of json files)
        train_unl_iter = self.get_iterator('sub')
        labeled_batch = train_iter.next()
        unlabeled_batch = train_unl_iter.next()

        # Yield process information for calculating current consistency weight
        epoch = train_iter.epoch
        process_info = {'epoch': epoch, 'current_position': train_iter.current_position, 'batch_len': train_iter.len}

        # self.iteration += 1 # Increase may result in early report, which is done in other place automatically.
        labeled_x = _recursive_to(labeled_batch, self.device)
        unlabeled_x = _recursive_to(unlabeled_batch, self.device)
        is_new_epoch = train_iter.epoch != epoch
        # When the last minibatch in the current epoch is given,
        # gradient accumulation is turned off in order to evaluate the model
        # on the validation set in every epoch.
        # see details in https://github.com/espnet/espnet/pull/1388
        # Compute the loss at this time step and accumulate it
        if self.ngpu == 0:
            loss = self.model(*labeled_x, *unlabeled_x, process_info)
        else:
            # apex does not support torch.nn.DataParallel
            loss = data_parallel(self.model, (*labeled_x, *unlabeled_x, process_info), range(self.ngpu))
        loss = loss.mean() / self.accum_grad
        loss.backward()

        # learning rate cosine rampdown for SGD optimizer
        if epoch > self.cosine_rampdown_starts:
            for p in optimizer.param_groups:
                p["lr"] *= cosine_rampdown(epoch - self.cosine_rampdown_starts,
                            self.cosine_rampdown_ends - self.cosine_rampdown_starts)
            logging.info("learning rate decayed to " + str(p["lr"]))

        # gradient noise injection
        if self.grad_noise:
            from espnet.asr.asr_utils import add_gradient_noise
            add_gradient_noise(self.model, self.iteration, duration=100, eta=1.0, scale_factor=0.55)
        loss.detach()  # Truncate the graph
        # update parameters
        self.forward_count += 1
        if not is_new_epoch and self.forward_count != self.accum_grad:
            return
        self.forward_count = 0
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_threshold)
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()
            global_step = epoch * train_iter.len + train_iter.current_position
            if epoch < self.consistency_rampup_ends:
                update_ema_variables(self.model.enc, self.model.ema_enc, self.ema_pre_decay, global_step)
            else:
                update_ema_variables(self.model.enc, self.model.ema_enc, self.ema_post_decay, global_step)
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

    def __init__(self, subsampling_factor=1, dtype=torch.float32):
        """Construct a CustomConverter object."""
        self.subsampling_factor = subsampling_factor
        self.ignore_id = -1
        self.dtype = dtype

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

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        # currently only support real number
        if xs[0].dtype.kind == "c":
            xs_pad_real = pad_list(
                [torch.from_numpy(x.real).float() for x in xs], 0
            ).to(device, dtype=self.dtype)
            xs_pad_imag = pad_list(
                [torch.from_numpy(x.imag).float() for x in xs], 0
            ).to(device, dtype=self.dtype)
            # Note(kamo):
            # {'real': ..., 'imag': ...} will be changed to ComplexTensor in E2E.
            # Don't create ComplexTensor and give it E2E here
            # because torch.nn.DataParellel can't handle it.
            xs_pad = {"real": xs_pad_real, "imag": xs_pad_imag}
        else:
            xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(
                device, dtype=self.dtype
            )

        ilens = torch.from_numpy(ilens).to(device)
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

        return xs_pad, ilens, ys_pad


class CustomConverterMulEnc(object):
    """Custom batch converter for Pytorch in multi-encoder case.

    Args:
        subsampling_factors (list): List of subsampling factors for each encoder.
        dtype (torch.dtype): Data type to convert.

    """

    def __init__(self, subsamping_factors=[1, 1], dtype=torch.float32):
        """Initialize the converter."""
        self.subsamping_factors = subsamping_factors
        self.ignore_id = -1
        self.dtype = dtype
        self.num_encs = len(subsamping_factors)

    def __call__(self, batch, device=torch.device("cpu")):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple( list(torch.Tensor), list(torch.Tensor), torch.Tensor)

        """
        # batch should be located in list
        assert len(batch) == 1
        xs_list = batch[0][: self.num_encs]
        ys = batch[0][-1]

        # perform subsampling
        if np.sum(self.subsamping_factors) > self.num_encs:
            xs_list = [
                [x[:: self.subsampling_factors[i], :] for x in xs_list[i]]
                for i in range(self.num_encs)
            ]

        # get batch of lengths of input sequences
        ilens_list = [
            np.array([x.shape[0] for x in xs_list[i]]) for i in range(self.num_encs)
        ]

        # perform padding and convert to tensor
        # currently only support real number
        xs_list_pad = [
            pad_list([torch.from_numpy(x).float() for x in xs_list[i]], 0).to(
                device, dtype=self.dtype
            )
            for i in range(self.num_encs)
        ]

        ilens_list = [
            torch.from_numpy(ilens_list[i]).to(device) for i in range(self.num_encs)
        ]
        # NOTE: this is for multi-task learning (e.g., speech translation)
        ys_pad = pad_list(
            [
                torch.from_numpy(np.array(y[0]) if isinstance(y, tuple) else y).long()
                for y in ys
            ],
            self.ignore_id,
        ).to(device)

        return xs_list_pad, ilens_list, ys_pad


def train(args):
    """Train with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)
    if args.num_encs > 1:
        args = format_mulenc_args(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning("cuda is not available")

    # get input and output dimension info
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]
    utts = list(valid_json.keys())
    idim_list = [
        int(valid_json[utts[0]]["input"][i]["shape"][-1]) for i in range(args.num_encs)
    ]
    odim = int(valid_json[utts[0]]["output"][0]["shape"][-1])
    for i in range(args.num_encs):
        logging.info("stream{}: input dims : {}".format(i + 1, idim_list[i]))
    logging.info("#output dims: " + str(odim))

    # specify semi-supervised method
    assert 0.0 <= args.mixup_alpha <= 1.0, "mixup-alpha should be [0.0, 1.0]"
    if args.mixup_alpha == 0.0:
        semi_mode = "MT"
        logging.info("Pure Mean-Teacher mode")
    else:
        semi_mode = "ICT"
        logging.info("Interpolation Consistency Training mode")

    if (args.enc_init is not None or args.dec_init is not None) and args.num_encs == 1:
        model = load_trained_modules(idim_list[0], odim, args)
    else:
        model_class = dynamic_import(args.model_module)
        model = model_class(
            idim_list[0] if args.num_encs == 1 else idim_list, odim, args
        )

    assert isinstance(model, ASRInterface)

    if args.rnnlm is not None:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(len(args.char_list), rnnlm_args.layer, rnnlm_args.unit)
        )
        torch_load(args.rnnlm, rnnlm)
        model.rnnlm = rnnlm

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + "/model.json"
    with open(model_conf, "wb") as f:
        logging.info("writing a model config file to " + model_conf)
        f.write(
            json.dumps(
                (idim_list[0] if args.num_encs == 1 else idim_list, odim, vars(args)),
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
        if args.num_encs > 1:
            # TODO(ruizhili): implement data parallel for multi-encoder setup.
            raise NotImplementedError(
                "Data parallel is not supported for multi-encoder setup."
            )

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
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
    elif args.opt == "noam":
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt

        optimizer = get_std_opt(
            model, args.adim, args.transformer_warmup_steps, args.transformer_lr
        )
    elif args.opt == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0008, alpha=0.95)
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.95, nesterov=True)
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

        from espnet.nets.pytorch_backend.ctc import CTC

        amp.register_float_function(CTC, "loss_fn")
        amp.init()
        logging.warning("register ctc as float function")
    else:
        use_apex = False

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # Setup a converter
    if args.num_encs == 1:
        converter = CustomConverter(subsampling_factor=model.subsample[0], dtype=dtype)
    else:
        converter = CustomConverterMulEnc(
            [i[0] for i in model.subsample_list], dtype=dtype
        )

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    assert 0.0 < args.utt_using_ratio < 1.0, "utt-using-ratio should not be 0 or 1"

    train_labeled_json_path = args.train_json.replace('.json', '_labeled_{}.json'.
                                                                  format(int(args.utt_using_ratio * 100)))
    train_unlabeled_json_path = args.train_json.replace('.json', '_unlabeled_{}.json'.
                                                        format(100 - int(args.utt_using_ratio * 100)))
    if os.path.exists(train_labeled_json_path):
        with open(train_labeled_json_path, 'rb') as f:
            train_labeled_json = json.load(f)['utts']
        with open(train_unlabeled_json_path, 'rb') as f:
            train_unlabeled_json = json.load(f)['utts']
    else:
        # split json for each task
        split_point = [int(len(train_json) * args.utt_using_ratio)]
        train_labeled_json = dict(list(train_json.items())[:split_point[0]])
        train_unlabeled_json = dict(list(train_json.items())[split_point[0]:])
        with codecs.open(train_labeled_json_path, 'w+', encoding='utf8') as f:
            json.dump({'utts': train_labeled_json}, f, indent=4, sort_keys=True,
                      ensure_ascii=False, separators=(',', ': '))
        with codecs.open(train_unlabeled_json_path, 'w+', encoding='utf8') as f:
            json.dump({'utts': train_unlabeled_json}, f, indent=4, sort_keys=True,
                      ensure_ascii=False, separators=(',', ': '))

    valid_labeled_json_path = args.valid_json.replace('.json', '_labeled_{}.json'.
                                                      format(int(args.utt_using_ratio * 100)))
    valid_unlabeled_json_path = args.valid_json.replace('.json', '_unlabeled_{}.json'.
                                                        format(100 - int(args.utt_using_ratio * 100)))
    if os.path.exists(valid_labeled_json_path):
        with open(valid_labeled_json_path, 'rb') as f:
            valid_labeled_json = json.load(f)['utts']
        with open(valid_unlabeled_json_path, 'rb') as f:
            valid_unlabeled_json = json.load(f)['utts']
    else:
        # split json for each task
        split_point = [int(len(valid_json) * args.utt_using_ratio)]
        valid_labeled_json = dict(list(valid_json.items())[:split_point[0]])
        valid_unlabeled_json = dict(list(valid_json.items())[split_point[0]:])
        with codecs.open(valid_labeled_json_path, 'w+', encoding='utf8') as f:
            json.dump({'utts': valid_labeled_json}, f, indent=4, sort_keys=True,
                      ensure_ascii=False, separators=(',', ': '))
        with codecs.open(valid_unlabeled_json_path, 'w+', encoding='utf8') as f:
            json.dump({'utts': valid_unlabeled_json}, f, indent=4, sort_keys=True,
                      ensure_ascii=False, separators=(',', ': '))

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # make minibatch list (variable length)
    train = make_batchset(train_labeled_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          shortest_first=use_sortagrad,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout,
                          iaxis=0, oaxis=0)
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout,
                          iaxis=0, oaxis=0)
    ul_train = make_batchset(train_unlabeled_json, args.batch_size,
                              args.maxlen_in, args.maxlen_out, args.minibatches,
                              min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                              shortest_first=use_sortagrad,
                              count=args.batch_count,
                              batch_bins=args.batch_bins,
                              batch_frames_in=args.batch_frames_in,
                              batch_frames_out=args.batch_frames_out,
                              batch_frames_inout=args.batch_frames_inout,
                              iaxis=0, oaxis=0)
    ul_valid = make_batchset(valid_unlabeled_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout,
                          iaxis=0, oaxis=0)
    load_tr = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True}  # Switch the mode of preprocessing
    )
    load_cv = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )
    load_ul_tr = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True}  # Switch the mode of preprocessing
    )
    load_ul_cv = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
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
    # Add splited data iteration for training
    ul_train_iter = ChainerDataLoader(
        dataset=TransformDataset(ul_train, lambda data: converter([load_ul_tr(data)])),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0],
        num_workers=args.n_iter_processes,
    )
    ul_valid_iter = ChainerDataLoader(
        dataset=TransformDataset(ul_valid, lambda data: converter([load_ul_cv(data)])),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0],
        num_workers=args.n_iter_processes,
    )

    # Set up ICT related arguments
    ICT_args = {"consistency_rampup_ends": args.consistency_rampup_ends,
                "cosine_rampdown_starts": args.cosine_rampdown_starts,
                "cosine_rampdown_ends": args.cosine_rampdown_ends,
                "ema_pre_decay": args.ema_pre_decay,
                "ema_post_decay": args.ema_post_decay
    }

    # Set up a trainer
    updater = CustomUpdater(
        model,
        args.grad_clip,
        {"main": train_iter, "sub": ul_train_iter},
        optimizer,
        device,
        args.ngpu,
        ICT_args,
        args.grad_noise,
        args.accum_grad,
        use_apex=use_apex,
    )
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    if use_sortagrad:
        trainer.extend(
            ShufflingEnabler([train_iter]),
            trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, "epoch"),
        )

    # Resume from a snapshot
    if args.resume:
        logging.info("resumed from %s" % args.resume)
        torch_resume(args.resume, trainer)

    # Evaluate the model with the test dataset for each epoch
    # TODO: custom evaluator
    if args.save_interval_iters > 0:
        trainer.extend(
            CustomEvaluator(model, {"main": valid_iter, "sub": ul_valid_iter}, reporter, device, args.ngpu),
            trigger=(args.save_interval_iters, "iteration"),
        )
    else:
        trainer.extend(
            CustomEvaluator(model, {"main": valid_iter, "sub": ul_valid_iter}, reporter, device, args.ngpu)
        )

    # Make a plot for training and validation values
    trainer.extend(
        extensions.PlotReport(
            [
                "main/loss",
                "validation/main/loss",
                "main/loss_ce",
                "validation/main/loss_ce",
                "main/loss_mse",
                "validation/main/loss_mse",
            ],
            "epoch",
            file_name="loss.png",
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ["main/teacher_acc", "validation/main/teacher_acc"]
            + (["main/student_acc", "validation/main/student_acc"] if args.show_student_model_acc else []),
            "epoch", file_name="acc.png"
        )
    )

    # Save best models
    trainer.extend(
        snapshot_object(model, "model.loss.best"),
        trigger=training.triggers.MinValueTrigger("validation/main/loss"),
    )
    trainer.extend(
        snapshot_object(model, "model.acc.best"),
        trigger=training.triggers.MaxValueTrigger("validation/main/student_acc"),
    )

    # save snapshot which contains model and optimizer states
    if args.save_interval_iters > 0:
        trainer.extend(
            torch_snapshot(filename="snapshot.iter.{.updater.iteration}"),
            trigger=(args.save_interval_iters, "iteration"),
        )
    else:
        trainer.extend(torch_snapshot(), trigger=(1, "epoch"))

    # epsilon decay in the optimizer
    if args.opt == "adadelta":
        if args.criterion == "acc":
            trainer.extend(
                restore_snapshot(
                    model, args.outdir + "/model.acc.best", load_fn=torch_load
                ),
                trigger=CompareValueTrigger(
                    "validation/main/teacher_acc",
                    lambda best_value, current_value: best_value > current_value,
                ),
            )
            trainer.extend(
                adadelta_eps_decay(args.eps_decay),
                trigger=CompareValueTrigger(
                    "validation/main/teacher_acc",
                    lambda best_value, current_value: best_value > current_value,
                ),
            )
        elif args.criterion == "loss":
            trainer.extend(
                restore_snapshot(
                    model, args.outdir + "/model.loss.best", load_fn=torch_load
                ),
                trigger=CompareValueTrigger(
                    "validation/main/loss",
                    lambda best_value, current_value: best_value < current_value,
                ),
            )
            trainer.extend(
                adadelta_eps_decay(args.eps_decay),
                trigger=CompareValueTrigger(
                    "validation/main/loss",
                    lambda best_value, current_value: best_value < current_value,
                ),
            )

    # Write a log of evaluation statistics for each epoch
    trainer.extend(
        extensions.LogReport(trigger=(args.report_interval_iters, "iteration"))
    )
    report_keys = [
                      "epoch",
                      "iteration",
                      "main/loss",
                      "main/loss_ce",
                      "main/loss_mse",
                      "validation/main/loss",
                      "validation/main/loss_ce",
                      "validation/main/loss_mse",
                      "main/teacher_acc",
                      "validation/main/teacher_acc",
                      "elapsed_time",
                  ] + ["main/student_acc", "validation/main/student_acc"] if args.show_student_model_acc else []
    if args.opt == "adadelta":
        trainer.extend(
            extensions.observe_value(
                "eps",
                lambda trainer: trainer.updater.get_optimizer("main").param_groups[0][
                    "eps"
                ],
            ),
            trigger=(args.report_interval_iters, "iteration"),
        )
        report_keys.append("eps")
    trainer.extend(
        extensions.PrintReport(report_keys),
        trigger=(args.report_interval_iters, "iteration"),
    )

    trainer.extend(extensions.ProgressBar(update_interval=args.report_interval_iters))
    set_early_stop(trainer, args)

    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        trainer.extend(
            TensorboardLogger(SummaryWriter(args.tensorboard_dir), None),
            trigger=(args.report_interval_iters, "iteration"),
        )
    # Run the training
    trainer.run()
    check_early_stop(trainer, args.epochs)

# TODO: In which way are we going to show decoded results?
def recog(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)
    model.recog_args = args

    if args.streaming_mode and "transformer" in train_args.model_module:
        raise NotImplementedError("streaming mode for transformer is not implemented")

    # read rnnlm
    rnnlm = None

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info("gpu id: " + str(gpu_id))
        model.cuda()

    # read json data
    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]
    new_js = {}

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    ark_file = open(args.result_ark,'wb')
    if args.batchsize == 0:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info("(%d/%d) decoding " + name, idx, len(js.keys()))
                batch = [(name, js[name])]
                feat = load_inputs_and_targets(batch)
                feat = (
                    feat[0][0]
                    if args.num_encs == 1
                    else [feat[idx][0] for idx in range(model.num_encs)]
                )
                hyps = model.recognize(
                    feat, args, train_args.char_list, rnnlm
                )
                # TODO: is there any way to overwrite decoding results into new js?
                hyps = hyps.squeeze(1)
                hyps = hyps.data.numpy()
                write_mat(ark_file, hyps, key=name)
