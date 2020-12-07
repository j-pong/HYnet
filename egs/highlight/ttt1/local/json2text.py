#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 J-pong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import json
import logging

from espnet.utils.cli_utils import get_commandline_args


def get_parser():
    parser = argparse.ArgumentParser(
        description="convert liner json to text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("json", type=str, help="json files")
    parser.add_argument("text", type=str, help="text")
    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()
    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    logging.info("reading %s", args.json)
    with codecs.open(args.json, "r", encoding="utf-8") as f:
        js = json.load(f)

    txt = codecs.open(args.text, "w", encoding="utf-8")

    for idx, j in enumerate(js):
        # txt.write(j['paragraph'] + "\n")
        txt.write(str(idx) + ' ' + j['paragraph'] + "\n")
        # txt.write(str(idx) + '   ' + j['paragraph'] + '  ' + j['sentence'] + "\n")
