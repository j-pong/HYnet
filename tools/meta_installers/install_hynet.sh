#!/bin/bash

# The espent editable import to HYnet
. ./activate_python.sh && python3 -m pip install -e .

touch hynet.done
