
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
from ..utils.model_utils import load_checkpoint,remove_lightning_load_state

def load_model(*args,**kwargs):

    # -- defaults changed by noise version --
    noise_version = optional(kwargs,'noise_version',"noise")
    if noise_version == "noise":
        default_modulator = True
        default_depth = [1, 2, 8, 8, 2, 8, 8, 2, 1]
        # default_modulator = False
        # default_depth = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    elif noise_version == "blur":
        default_modulator = True
        default_depth = [1, 2, 8, 8, 2, 8, 8, 2, 1]
    else:
        raise ValueError(f"Uknown noise version [{noise_version}]")

    # -- get cfg --
    nchnls = optional(kwargs,'nchnls',3)
    input_size = optional(kwargs,'input_size',128)
    depths = optional(kwargs,'input_size',default_depth)
    device = optional(kwargs,'device','cuda:0')

    # -- other configs --
    embed_dim = optional(kwargs,'embed_dim',32)
    # embed_dim = optional(kwargs,'embed_dim',44)
    win_size = optional(kwargs,'win_size',8)
    mlp_ratio = optional(kwargs,'mlp_ratio',4)
    qkv_bias = optional(kwargs,'qkv_bias',True)
    token_projection = optional(kwargs,'token_projection','linear')
    token_mlp = optional(kwargs,'token_mlp','leff')
    modulator = optional(kwargs,'modulator',default_modulator)
    cross_modulator = optional(kwargs,'cross_modulator',False)

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
                    ws=ws,wt=wt,k=k,sb=sb,
                    modulator=modulator,cross_modulator=cross_modulator)
    model = model.to(device)

    # -- load weights --
    # model_sigma = select_sigma(data_sigma)
    fdir = Path(__file__).absolute().parents[0] / "../../../" # parent of "./lib"
    lit = False
    if noise_version == "noise":
        # state_fn = "output/checkpoints/bc1b491e-e536-43a6-9261-88de75c17deb-epoch=15-val_loss=1.58e-04.ckpt"
        # state_fn = "output/checkpoints/44006e54-ddb2-4776-8cb0-e86edc464370-epoch=09-val_loss=1.55e-04.ckpt"
        # lit = True
        state_fn = fdir / "models/Uformer_sidd_B.pth"
        lit = False
    elif noise_version == "blur":
        state_fn = fdir / "models/Uformer_gopro_B.pth"
    else:
        raise ValueError(f"Uknown noise_version [{noise_version}]")
    assert os.path.isfile(str(state_fn))
    # model_state = th.load(str(state_fn))

    # -- fill weights --
    if lit is False:
        load_checkpoint(model,state_fn)
    else:
        state = th.load(state_fn)['state_dict']
        remove_lightning_load_state(state)
        model.load_state_dict(state)

    # load_checkpoint(model,model_state)
    # model.load_state_dict(model_state['state_dict'])

    # -- eval mode as default --
    model.eval()

    return model

