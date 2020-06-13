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
        images = self.get_images()

        for idx, img in enumerate(images):
            filename = "%s/%s.ep.{.updater.epoch}.png" % (
                self.outdir,
                self.data[idx][0],
            )
            img = self.get_attention_weight(idx, img)
            # np_filename = "%s/%s.ep.{.updater.epoch}.npy" % (
            #     self.outdir,
            #     self.data[idx][0],
            # )
            # np.save(np_filename.format(trainer), att_w)
            self._plot_and_save_image(img, filename.format(trainer))

    def get_images(self):
        batch = self.converter([self.transform(self.data)], self.device)
        if isinstance(batch, tuple):
            att_ws = self.vis_fn(*batch)
        else:
            att_ws = self.vis_fn(**batch)
        return att_ws

    def draw_image(self, img):
        import matplotlib.pyplot as plt

        img = img.astype(np.float32)
        if len(img.shape) == 3:
            for h, aw in enumerate(img, 1):
                plt.subplot(1, len(img), h)
                plt.imshow(aw, aspect="auto")
                plt.xlabel("Encoder Index")
                plt.ylabel("Decoder Index")
        else:
            plt.imshow(img, aspect="auto")
            plt.xlabel("Encoder Index")
            plt.ylabel("Decoder Index")
        plt.tight_layout()
        return plt

    def _plot_and_save_image(self, att_w, filename):
        plt = self.draw_image(att_w)
        plt.savefig(filename)
        plt.close()
