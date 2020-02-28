#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import json

import torch

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from moneynet.nets.simnn import Net
from moneynet.utils.datasets.pikachu_dataset import Pikachu


class Reporter(object):
    def __init__(self, args):
        self.outdir = args.outdir
        self.report_dict = {}

        self.report_buffer = {}

    def report_image(self, keys, epoch):
        # parsing images
        images = []
        for key in keys:
            image = self.report_dict[key]
            assert len(np.shape(image)) == 2, 'reported image should has dim == 2'
            images.append(image.T)
        num_image = len(images)

        fig = plt.Figure()
        fig.suptitle('{}'.format(self.report_dict['loss']))
        for i, image in enumerate(images):
            # one column setting
            ax = fig.add_subplot(num_image, 1, i + 1)
            ax.imshow(image, aspect='auto')

        sav_dir = os.path.join(self.outdir, 'images')
        if not os.path.exists(sav_dir):
            os.makedirs(sav_dir)
        filename = 'epoch{}_images.png'.format(epoch)
        fig.savefig(os.path.join(sav_dir, filename))

    def report_plot(self, keys, epoch):
        # parsing images
        scalars = []
        for key in keys:
            scalar = self.report_dict[key]
            if len(np.shape(scalar)) > 1:
                for i in range(np.shape(scalar)[-1]):
                    scalars.append(scalar[:, i])
            else:
                scalars.append(scalar)
        num_scalar = len(scalars)

        fig = plt.Figure()
        for i, scalar in enumerate(scalars):
            # one column setting
            ax = fig.add_subplot(num_scalar, 1, i + 1)
            ax.plot(range(len(scalar)), scalar)
            ax.grid()

        sav_dir = os.path.join(self.outdir, 'images')
        if not os.path.exists(sav_dir):
            os.makedirs(sav_dir)
        filename = 'epoch{}_scalars.png'.format(epoch)
        fig.savefig(os.path.join(sav_dir, filename))

    def report_plot_buffer(self, keys, epoch):
        for key in keys:
            scalar = self.report_dict[key]
            try:
                self.report_buffer[key].append(scalar)
            except:
                self.report_buffer[key] = [scalar]

        for key in keys:
            fig = plt.Figure()
            ax = fig.add_subplot(1, 1, 1)
            fig.suptitle('epoch : {}'.format(epoch))
            ax.plot(self.report_buffer[key], marker='o')
            ax.grid()

            filename = '{}.png'.format(key)
            fig.savefig(os.path.join(self.outdir, filename))

    def add_report_attribute(self, attribute):
        for key in attribute.keys():
            self.report_dict[key] = attribute[key]


class Updater(object):
    def __init__(self, args, train_loader, optimizer, device, model, reporter):
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.model = model
        self.device = device
        self.model = model
        self.reporter = reporter

        self.grad_clip = args.grad_clip
        self.accum_grad = args.accum_grad

        self.forward_count = 0

    def pretrain_core(self):
        for samples in self.train_loader:
            data = samples['input'][0].to(self.device)

            loss = self.model.pretrain_forward(data)
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
        self.reporter.report_dict['loss'] = float(loss)

    def train_core(self):
        for samples in self.train_loader:
            data = samples['input'][0].to(self.device)
            target = samples['target'][0].to(self.device)

            loss, pred = self.model(data, target)
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

        # ToDo(j-pong): Add reporter attribute but just epoch mode
        self.reporter.report_dict['loss'] = float(loss)
        self.reporter.report_dict['pred'] = pred.view(-1, data.size(1), data.size(2))[0].detach().cpu().numpy()
        self.reporter.report_dict['target'] = target[0].detach().cpu().numpy()
        self.reporter.report_dict['fname'] = samples['fname'][0]


def train(args):
    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get dataset
    dataset = Pikachu(args=args)

    # reverse input and output dimension
    idim, odim = map(int, dataset.__dims__())
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

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

    # specify model architecture
    reporter = Reporter(args)
    model = Net(idim, odim, args, reporter)
    logging.info(model)

    # check the use of multi-gpu
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        if args.batch_size != 0:
            logging.warning('batch size is automatically increased (%d -> %d)' % (
                args.batch_size, args.batch_size * args.ngpu))
            args.batch_size *= args.ngpu

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

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=6, pin_memory=True)

    updater = Updater(args, train_loader, optimizer, device, model, reporter)

    # Training dataset
    model.train()

    for epoch in tqdm(range(args.epochs)):
        updater.pretrain_core()
        if (epoch + 1) % args.low_interval_epochs == 0:
            reporter.report_plot_buffer(keys=['loss'], epoch=epoch + 1)

    for epoch in tqdm(range(args.epochs)):
        updater.train_core()
        if (epoch + 1) % args.high_interval_epochs == 0:
            reporter.report_image(keys=['target', 'pred', 'disentangle'], epoch=epoch + 1)
            reporter.report_plot(keys=['augs_p', 'augs_sim'], epoch=epoch + 1)
        if (epoch + 1) % args.low_interval_epochs == 0:
            reporter.report_plot_buffer(keys=['loss'], epoch=epoch + 1)
