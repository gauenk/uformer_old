"""

Test versions of Uformer to differences in output due to code modifications.

"""

# -- misc --
import sys,tqdm,pytest,math,random
from pathlib import Path

# -- dict data --
import copy
from easydict import EasyDict as edict

# -- vision --
from PIL import Image

# -- testing --
import unittest
import tempfile

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- data --
import data_hub

# -- package imports [to test] --
import dnls # supporting
from torchvision.transforms.functional import center_crop

# -- package imports [to test] --
import uformer
from uformer.utils.gpu_mem import print_gpu_stats,print_peak_gpu_stats

# -- check if reordered --
from scipy import optimize
MAX_NFRAMES = 85
DATA_DIR = Path("./data/")
SAVE_DIR = Path("./output/tests/test_denose_rgb/")
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)

def set_seed(seed):
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)
    # th.use_deterministic_algorithms(True)

def pytest_generate_tests(metafunc):
    seed = 123
    set_seed(seed)
    # test_lists = {"ps":[3],"stride":[2],"dilation":[2],
    #               "top":[3],"btm":[57],"left":[7],"right":[57]}
    test_lists = {"sigma":[50.]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# -->  Test original vs refactored code base  <--
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# @pytest.mark.skip()
def test_original_refactored(sigma):

    # -- params --
    device = "cuda:0"
    vid_set = "sidd_rgb"
    vid_name = "00"
    verbose = False

    # -- setup cfg --
    cfg = edict()
    cfg.dname = vid_set
    cfg.vid_name = vid_name

    # -- video --
    data,loaders = data_hub.sets.load(cfg)
    groups = data.val.groups
    indices = [i for i,g in enumerate(groups) if cfg.vid_name == g]
    index = indices[0]

    # -- unpack --
    sample = data.val[index]
    noisy,clean = sample['noisy'],sample['clean']
    noisy,clean = noisy.to(device),clean.to(device)
    vid_frames = sample['fnums']
    noisy /= 255.

    # -- original exec --
    og_model = uformer.original.load_model(sigma)
    with th.no_grad():
        deno_og = og_model(noisy.clone()).detach()

    # -- each version --
    for ref_version in ["ref"]: #["original","ref"]:

        # -- refactored exec --
        ref_model = uformer.refactored.load_model(sigma,mode=ref_version)
        with th.no_grad():
            deno_ref = ref_model(noisy).detach()

        # -- viz --
        if verbose:
            print(deno_og[0,0,:3,:3])
            print(deno_ref[0,0,:3,:3])

        # -- test --
        error = th.sum((deno_og - deno_ref)**2).item()
        if verbose: print("error: ",error)
        assert error < 1e-15
