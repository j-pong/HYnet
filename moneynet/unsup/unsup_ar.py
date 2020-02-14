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
from moneynet.utils.pikachu_dataset import Pikachu


class Reporter(object):
    def __init__(self, args):
        self.outdir = args.outdir
        self.report_dict = {}

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
        filename = 'epoch{}.png'.format(epoch)
        fig.savefig(os.path.join(sav_dir, filename))

    def report_plot(self):
        pass

    def add_report_attribute(self, attribute):
        for key in attribute.keys():
            self.report_dict[key] = attribute[key]


def train_core(train_loader, optimizer, device, model, reporter):
    for samples in train_loader:
        data = samples['input'][0].to(device)
        target = samples['target'][0].to(device)

        optimizer.zero_grad()
        loss, pred = model(data, target)
        loss.backward()
        optimizer.step()

    # ToDo(j-pong): Add reporter attribute but just epoch mode
    reporter.report_dict['loss'] = float(loss)
    reporter.report_dict['pred'] = pred.view(-1, data.size(1), data.size(2))[0].detach().cpu().numpy()
    reporter.report_dict['target'] = target[0].detach().cpu().numpy()
    reporter.report_dict['fname'] = samples['fname'][0]


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
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # Training dataset
    model.train()
    for epoch in tqdm(range(args.epochs)):
        train_core(train_loader, optimizer, device, model, reporter)
        if (epoch + 1) % 100 == 0:
            reporter.report_image(keys=['target', 'pred'], epoch=epoch + 1)
