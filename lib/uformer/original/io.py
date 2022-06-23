
# -- misc --
import sys,os,copy
from pathlib import Path

# -- torch --
import torch as th

# -- linalg --
import numpy as np

# -- modules --
from .model import Uformer

# -- misc imports --
from ..common import optional,select_sigma

def load_model(*args,**kwargs):

    # -- get cfg --
    device = optional(kwargs,'device','cuda:0')
    data_sigma = optional(kwargs,'sigma','cuda:0')

    # -- init model --
    model = Uformer().to(device)
    model.cuda()

    # -- load weights --
    model_sigma = select_sigma(data_sigma)
    fdir = Path(__file__).absolute().parents[0] / "../../../" # parent of "./lib"
    state_fn = fdir / "models/model_state_sigma_{}_c.pt".format(lidia_sigma)
    assert os.path.isfile(str(state_fn))
    model_state = th.load(str(state_fn))

    # -- fill weights --
    model.load_state_dict(model_state['state_dict'])

    return model
