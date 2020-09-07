import sys
import argparse
import logging

from typing import Sequence

import numpy as np
import cv2 as cv

from espnet2.tasks.abs_task import AbsTask, IteratorOptions, AbsIterFactory
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str_or_int
from espnet2.utils.types import str_or_none

from hynet.imgm.frontend.kpd import KeypointDetection
from hynet.imgm.frontend.match import TemplateMatching


class ImgmTask(AbsTask):
    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        group.add_argument(
            "--input_dir",
            type=str_or_none,
            default=None,
            help="",
        )

        group.add_argument(
            "--template_dir",
            type=str_or_none,
            default=None,
            help="",
        )

    @classmethod
    def main(cls, args: argparse.Namespace = None, cmd: Sequence[str] = None):
        if args is None:
            parser = cls.get_parser()
            args = parser.parse_args(cmd)
        if args.print_config:
            cls.print_config()
            sys.exit(0)

        streamer = cls.build_streaming_iterator(args=args)
        template = cv.imread(args.template_dir)

        tm = TemplateMatching()

        while streamer.isOpened():
            ret, img = streamer.read()

            if not ret:
                raise ValueError("Can't receive frame (stream end?). Exiting ...")
                break
            
            img_crop, score = tm.recognize(img, template)

            if tm.peak_detection(score):
                import matplotlib.pyplot as plt
                plt.clf()
                plt.subplot(221)
                plt.imshow(img)
                plt.subplot(222)
                plt.imshow(img_crop)
                plt.subplot(213)
                plt.imshow(template)
                plt.savefig("imgm.png")
                
        streamer.release()


    @classmethod
    def build_streaming_iterator(cls, args: argparse.Namespace = None):
        cap = cv.VideoCapture(args.input_dir)
        return cap
