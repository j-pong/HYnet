#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

# matplotlib related
import os

import matplotlib.pyplot as plt
import numpy as np
import copy

from chainer.training import extension

from espnet.asr import asr_utils

class PlotλcReport(extension.Extension):
    """Plot attention reporter.

    Args:
        att_vis_fn (espnet.nets.*_backend.e2e_asr.E2E.calculate_all_attentions):
            Function of attention visualization.
        data (list[tuple(str, dict[str, list[Any]])]): List json utt key items.
        outdir (str): Directory to save figures.
        converter (espnet.asr.*_backend.asr.CustomConverter): Function to convert data.
        device (int | torch.device): Device.
        reverse (bool): If True, input and output length are reversed.
        ikey (str): Key to access input (for ASR ikey="input", for MT ikey="output".)
        iaxis (int): Dimension to access input (for ASR iaxis=0, for MT iaxis=1.)
        okey (str): Key to access output (for ASR okey="input", MT okay="output".)
        oaxis (int): Dimension to access output (for ASR oaxis=0, for MT oaxis=0.)

    """

    def __init__(
        self,
        att_vis_fn,
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
        self.att_vis_fn = att_vis_fn
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
    
    def get_λc(self):
        batch = self.converter([self.transform(self.data)], self.device)
        if isinstance(batch, tuple):
            λc = self.att_vis_fn(*batch)
        elif isinstance(batch, dict):
            λc = self.att_vis_fn(**batch)
        return λc
    
    def __call__(self, trainer):
        """Plot and save image file of λcs matrix."""
        λc_s = self.get_λc()
        
        for idx, λc in enumerate(λc_s):
            filename = "%s/%s.ep.{.updater.epoch}.png" % (
                self.outdir,
                self.data[idx][0],
            )
            λc = self.get_λc(idx, λc)
            np_filename = "%s/%s.ep.{.updater.epoch}.npy" % (
                self.outdir,
                self.data[idx][0],
            )
            np.save(np_filename.format(trainer), λc)
            self._plot_and_save_λc(λc, filename.format(trainer))

    def _plot_and_save_λc(self, λc, filename, xtokens=None, ytokens=None):
        # dynamically import matplotlib due to not found error
        from matplotlib.ticker import MaxNLocator
        import os
    
        d = os.path.dirname(filename)
        if not os.path.exists(d):
            os.makedirs(d)
        w, h = plt.figaspect(1.0 / len(λc))
        fig = plt.Figure(figsize=(w * 2, h * 2))
        axes = fig.subplots(1, len(λc))
        if len(λc) == 1:
            axes = [axes]
        for ax, aw in zip(axes, λc):
            # plt.subplot(1, len(λc), h)
            ax.imshow(aw.astype(np.float32), aspect="auto")
            ax.set_xlabel("λc")
            ax.set_ylabel("Output")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            # Labels for major ticks
            if xtokens is not None:
                ax.set_xticks(np.linspace(0, len(xtokens) - 1, len(xtokens)))
                ax.set_xticks(np.linspace(0, len(xtokens) - 1, 1), minor=True)
                ax.set_xticklabels(xtokens + [""], rotation=40)
            if ytokens is not None:
                ax.set_yticks(np.linspace(0, len(ytokens) - 1, len(ytokens)))
                ax.set_yticks(np.linspace(0, len(ytokens) - 1, 1), minor=True)
                ax.set_yticklabels(ytokens + [""])
        fig.tight_layout()
        return fig


def savefig(plot, filename):
    plot.savefig(filename)
    plt.clf()


def plot_multi_head_attention(
    data,
    attn_dict,
    outdir,
    suffix="png",
    savefn=savefig,
    ikey="input",
    iaxis=0,
    okey="output",
    oaxis=0,
):
    """Plot multi head attentions.

    :param dict data: utts info from json file
    :param dict[str, torch.Tensor] attn_dict: multi head attention dict.
        values should be torch.Tensor (head, input_length, output_length)
    :param str outdir: dir to save fig
    :param str suffix: filename suffix including image type (e.g., png)
    :param savefn: function to save

    """
    for name, λcs in attn_dict.items():
        for idx, λc in enumerate(λcs):
            filename = "%s/%s.%s.%s" % (outdir, data[idx][0], name, suffix)
            dec_len = int(data[idx][1][okey][oaxis]["shape"][0])
            enc_len = int(data[idx][1][ikey][iaxis]["shape"][0])
            xtokens, ytokens = None, None
            if "encoder" in name:
                λc = λc[:, :enc_len, :enc_len]
                # for MT
                if "token" in data[idx][1][ikey][iaxis].keys():
                    xtokens = data[idx][1][ikey][iaxis]["token"].split()
                    ytokens = xtokens[:]
            elif "decoder" in name:
                if "self" in name:
                    λc = λc[:, : dec_len + 1, : dec_len + 1]  # +1 for <sos>
                else:
                    λc = λc[:, : dec_len + 1, :enc_len]  # +1 for <sos>
                    # for MT
                    if "token" in data[idx][1][ikey][iaxis].keys():
                        xtokens = data[idx][1][ikey][iaxis]["token"].split()
                # for ASR/ST/MT
                if "token" in data[idx][1][okey][oaxis].keys():
                    ytokens = ["<sos>"] + data[idx][1][okey][oaxis]["token"].split()
                    if "self" in name:
                        xtokens = ytokens[:]
            else:
                logging.warning("unknown name for shaping attention")
            fig = _plot_and_save_attention(λc, filename, xtokens, ytokens)
            savefn(fig, filename)


class PlotAttentionReport(asr_utils.PlotAttentionReport):
    def plotfn(self, *args, **kwargs):
        kwargs["ikey"] = self.ikey
        kwargs["iaxis"] = self.iaxis
        kwargs["okey"] = self.okey
        kwargs["oaxis"] = self.oaxis
        plot_multi_head_attention(*args, **kwargs)

    def __call__(self, trainer):
        attn_dict = self.get_attention_weights()
        suffix = "ep.{.updater.epoch}.png".format(trainer)
        self.plotfn(self.data, attn_dict, self.outdir, suffix, savefig)

    def get_attention_weights(self):
        batch = self.converter([self.transform(self.data)], self.device)
        if isinstance(batch, tuple):
            λcs = self.att_vis_fn(*batch)
        elif isinstance(batch, dict):
            λcs = self.att_vis_fn(**batch)
        return λcs

    def log_attentions(self, logger, step):
        def log_fig(plot, filename):
            from os.path import basename

            logger.add_figure(basename(filename), plot, step)
            plt.clf()

        attn_dict = self.get_attention_weights()
        self.plotfn(self.data, attn_dict, self.outdir, "", log_fig)
