#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import json

import torch
from torch.nn.parallel import data_parallel

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from moneynet.nets.simnn_unsup import Net
from moneynet.utils.datasets.pikachu_dataset import Pikachu


class Reporter(object):
    def __init__(self, args):
        self.outdir = args.outdir
        self.report_dict = {}
        self.report_buffer = {}

    @staticmethod
    def set_style(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.autoscale(enable=True, axis='x', tight=True)

    def report_image(self, keys, filename):
        # parsing images
        num_image = len(keys)
        fig = plt.figure()
        for i, key in enumerate(keys):
            # one column setting
            image = self.report_dict[key]
            assert len(np.shape(image)) == 2, 'reported image should has dim == 2'
            ax = fig.add_subplot(num_image, 1, i + 1)
            self.set_style(ax)
            ax.set_title(key)
            ax.imshow(image.T, aspect='auto', cmap='plasma')

        sav_dir = os.path.join(self.outdir, 'images')
        if not os.path.exists(sav_dir):
            os.makedirs(sav_dir)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        fig.savefig(os.path.join(sav_dir, filename))
        plt.close()

    def report_plot_buffer(self, keys, epoch, tag=None):
        for key in keys:
            scalar = self.report_dict[key]
            if tag is not None:
                try:
                    self.report_buffer[key + tag].append(scalar)
                except:
                    self.report_buffer[key + tag] = [scalar]
            else:
                try:
                    self.report_buffer[key].append(scalar)
                except:
                    self.report_buffer[key] = [scalar]

        for key in keys:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            self.set_style(ax)
            fig.suptitle('epoch : {}'.format(epoch))
            if tag is not None:
                key = key + tag
            ax.plot(self.report_buffer[key])
            ax.grid()

            filename = '{}.png'.format(key)
            fig.savefig(os.path.join(self.outdir, filename))
            plt.close()

    def add_report_attribute(self, attribute):
        for key in attribute.keys():
            self.report_dict[key] = attribute[key]


class Evaluator(object):
    def __init__(self, args, test_loader, device, model, reporter):
        self.test_loader = test_loader
        self.model = model
        self.device = device
        self.reporter = reporter
        # ToDo: reporter multi gpu not support
        # if args.ngpu is not None:
        #     self.ngpu = args.ngpu
        # elif device.type == "cpu":
        #     self.ngpu = 0
        # else:
        #     self.ngpu = 1
        self.ngpu = 1

    def evaluate_core(self):
        self.model.eval()
        with torch.no_grad():
            for samples in self.test_loader:
                self.reporter.report_dict['fname'] = samples['fname'][0]
                x = (samples['input'][0].to(self.device), samples['target'][0].to(self.device))
                if self.ngpu == 0:
                    self.model(*x)
                else:
                    data_parallel(self.model, x, range(self.ngpu))
        self.model.train()


class Updater(object):
    def __init__(self, args, train_loader, optimizer, device, model, reporter):
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.model = model
        self.device = device
        self.reporter = reporter
        self.ngpu = args.ngpu

        self.grad_clip = args.grad_clip
        self.accum_grad = args.accum_grad

        self.forward_count = 0

    def train_core(self):
        for samples in self.train_loader:
            x = (samples['input'][0].to(self.device), samples['target'][0].to(self.device))
            loss = data_parallel(self.model, x, range(self.ngpu)).mean() / self.accum_grad
            loss.backward()

            self.forward_count += 1
            if self.forward_count != self.accum_grad:
                continue
            self.forward_count = 0

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip)
            logging.info('grad norm={}'.format(grad_norm))
            if np.isnan(grad_norm):
                logging.warning('grad norm is nan. Do not update model.')
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()


def train(args):
    # set deterministic for pytorch
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # https://github.com/pytorch/pytorch/issues/6351

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get dataset
    train_dataset = Pikachu(args=args, train=True)
    eval_dataset = Pikachu(args=args, train=False)

    # reverse input and output dimension
    idim, odim = map(int, train_dataset.__dims__())
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # specify model architecture
    reporter = Reporter(args)
    model = Net(idim, odim, args, reporter)
    logging.info(model)

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to' + model_conf)
        f.write(json.dumps((idim, odim, vars(args)),
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    if args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, args.train_dtype)
    else:
        dtype = torch.float32
    model = model.to(device=device, dtype=dtype)

    # Setup an optimizer
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), args.lr, eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.ncpu,
                                               pin_memory=args.pin_memory)
    test_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=args.ncpu,
                                              pin_memory=args.pin_memory)
    updater = Updater(args, train_loader, optimizer, device, model, reporter)
    evaluator = Evaluator(args, test_loader, device, model, reporter)

    if args.resume:
        logging.info('resumed from %s' % args.resume)
        model.load_state_dict(torch.load(args.resume))

    # Training dataset
    model.train()
    for epoch in tqdm(range(args.epochs)):
        updater.train_core()
        if (epoch + 1) % args.high_interval_epochs == 0:
            evaluator.evaluate_core()
            sample_name = reporter.report_dict['fname'][0].split()[-1]
            filename = 'epoch{}_{}_sim.png'.format(epoch + 1, sample_name.split('.')[0])
            reporter.report_image(keys=['theta_opt', 'sim_opt'], filename=filename)
            filename = 'epoch{}_{}_attn.png'.format(epoch + 1, sample_name.split('.')[0])
            reporter.report_image(keys=['attn0', 'attn1', 'attn2', 'attn3', 'attn4'], filename=filename)
            filename = 'epoch{}_{}_xdis.png'.format(epoch + 1, sample_name.split('.')[0])
            reporter.report_image(keys=['x_dis0', 'x_dis1', 'x_dis2', 'x_dis3', 'x_dis4'], filename=filename)
            filename = 'epoch{}_{}_xres.png'.format(epoch + 1, sample_name.split('.')[0])
            reporter.report_image(keys=['x_res0', 'x_res1', 'x_res2', 'x_res3', 'x_res4'], filename=filename)
            filename = 'epoch{}_{}.png'.format(epoch + 1, sample_name.split('.')[0])
            reporter.report_image(keys=['target', 'pred_y', 'pred_x', 'res_x'], filename=filename)
            reporter.report_plot_buffer(keys=['loss', 'loss_x', 'loss_y'], epoch=epoch + 1, tag='_eval')
        if (epoch + 1) % args.low_interval_epochs == 0:
            reporter.report_plot_buffer(keys=['loss'], epoch=epoch + 1)
        if (epoch + 1) % args.save_interval_epochs == 0:
            filename = 'epoch{}.ckpt'.format(epoch + 1)
            torch.save(model.state_dict(), os.path.join(args.outdir, filename))
