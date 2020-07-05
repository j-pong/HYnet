#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 j-pong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import os

import numpy as np

from chainer.training import extension


class PlotImageReport(extension.Extension):
    def __init__(
            self,
            vis_fn,
            data,
            outdir,
            converter,
            transform,
            device,
            reverse=False,
            ikey="input",
            iaxis=0,
            okey="output",
            oaxis=0,
    ):
        self.vis_fn = vis_fn
        self.data = copy.deepcopy(data)
        self.outdir = outdir
        self.converter = converter
        self.transform = transform
        self.device = device
        self.reverse = reverse
        self.ikey = ikey
        self.iaxis = iaxis
        self.okey = okey
        self.oaxis = oaxis
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def __call__(self, trainer):
        ret, batch = self.get_ret()

        for key in ret.keys():
            for idx, img in enumerate(ret[key]):
                filename = "%s/%s:%s.ep.{.updater.epoch}.png" % (
                    self.outdir,
                    self.data[idx][0],
                    key
                )
                if key == 'out':
                    img = np.concatenate([batch[1][idx].transpose(0, 1).cpu().numpy()[0:1], img[0:1]], axis=0)
                self._plot_and_save_image(img, filename.format(trainer))

    def get_ret(self):
        batch = self.converter([self.transform(self.data)], self.device)
        if isinstance(batch, tuple):
            ret = self.vis_fn(*batch)
        else:
            ret = self.vis_fn(**batch)
        return ret, batch

    def draw_image(self, img):
        import matplotlib.pyplot as plt

        img = img.astype(np.float32)
        if len(img.shape) == 3:
            for h, im in enumerate(img, 1):
                plt.subplot(len(img), 1, h)
                plt.imshow(im.T, aspect="auto")
                plt.ylabel("dim")
                plt.xlabel("time")
        elif len(img.shape) == 1:
            plt.plot(img)
            plt.xlabel("time")
            plt.grid()
            plt.autoscale(enable=True, axis='x', tight=True)
        else:
            plt.imshow(img.T, aspect="auto")
            plt.xlabel("index")
            plt.ylabel("time")
        plt.tight_layout()
        return plt

    def _plot_and_save_image(self, img, filename):
        plt = self.draw_image(img)
        plt.savefig(filename)
        plt.close()
