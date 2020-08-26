#!/usr/bin/env python3
from hynet.tasks.imgr import ImgrTask


def get_parser():
    parser = ImgrTask.get_parser()
    return parser


def main(cmd=None):
    ImgrTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
