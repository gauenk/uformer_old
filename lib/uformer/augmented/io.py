
# -- misc --
import sys,os,copy
from pathlib import Path

# -- torch --
import torch as th

# -- linalg --
import numpy as np

# -- modules --
from .uformer import Uformer

# -- misc imports --
from ..common import optional,select_sigma
from ..utils.model_utils import load_checkpoint

def load_model(*args,**kwargs):

    # -- get cfg --
    nchnls = optional(kwargs,'nchnls',3)
    input_size = optional(kwargs,'input_size',128)
    depths = optional(kwargs,'input_size',[2, 2, 2, 2, 2, 2, 2, 2, 2])
    device = optional(kwargs,'device','cuda:0')

    # -- other configs --
    embed_dim = optional(kwargs,'embed_dim',32)
    # embed_dim = optional(kwargs,'embed_dim',44)
    win_size = optional(kwargs,'win_size',8)
    mlp_ratio = optional(kwargs,'mlp_ratio',4)
    qkv_bias = optional(kwargs,'qkv_bias',True)
    token_projection = optional(kwargs,'token_projection','linear')
    token_mlp = optional(kwargs,'token_mlp','leff')

    # -- relevant configs --
    fwd_mode = optional(kwargs,'fwd_mode',"dnls_k")
    stride = optional(kwargs,'stride',None)
    ws = optional(kwargs,'ws',-1)
    wt = optional(kwargs,'wt',0)
    k = optional(kwargs,'k',-1)
    sb = optional(kwargs,'sb',None)

    # -- init model --
    model = Uformer(img_size=input_size, in_chans=nchnls, embed_dim=embed_dim,
                    depths=depths, win_size=win_size, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, token_projection=token_projection,
                    token_mlp=token_mlp,stride=stride,fwd_mode=fwd_mode,
                    ws=ws,wt=wt,k=k,sb=sb)
    model = model.to(device)

    # -- load weights --
    # model_sigma = select_sigma(data_sigma)
    fdir = Path(__file__).absolute().parents[0] / "../../../" # parent of "./lib"
    state_fn = fdir / "models/uformer32_denoising_sidd.pth"
    assert os.path.isfile(str(state_fn))
    # model_state = th.load(str(state_fn))

    # -- fill weights --
    load_checkpoint(model,state_fn)
    # load_checkpoint(model,model_state)
    # model.load_state_dict(model_state['state_dict'])

    # -- eval mode as default --
    model.eval()

    return model

