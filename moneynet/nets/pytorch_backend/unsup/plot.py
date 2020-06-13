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
        ret = self.get_ret()

        for idx, key in enumerate(ret.keys()):
            filename = "%s/%s:%s.ep.{.updater.epoch}.png" % (
                self.outdir,
                self.data[idx][0],
                key
            )
            self._plot_and_save_image(ret[key], filename.format(trainer))

    def get_ret(self):
        batch = self.converter([self.transform(self.data)], self.device)
        if isinstance(batch, tuple):
            ret = self.vis_fn(*batch)
        else:
            ret = self.vis_fn(**batch)
        return ret

    def draw_image(self, img):
        import matplotlib.pyplot as plt

        img = img.astype(np.float32)
        if len(img.shape) == 3:
            for h, aw in enumerate(img, 1):
                plt.subplot(len(img), 1, h)
                plt.imshow(aw.T, aspect="auto")
                plt.ylabel("feature_dim")
                plt.xlabel("time")
        else:
            plt.imshow(img, aspect="auto")
            plt.xlabel("index")
            plt.ylabel("time")
        plt.tight_layout()
        return plt

    def _plot_and_save_image(self, att_w, filename):
        plt = self.draw_image(att_w)
        plt.savefig(filename)
        plt.close()
