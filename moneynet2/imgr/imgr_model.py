from contextlib import contextmanager
from distutils.version import LooseVersion

import torch
from torch import nn

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

class ImgrModel(nn.model):
    def __init__(self, args):