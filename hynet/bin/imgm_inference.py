#!/usr/bin/env python3
from hynet.tasks.imgm import ImgmTask


def get_parser():
    parser = ImgmTask.get_parser()
    return parser


def main(cmd=None):
    ImgmTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
