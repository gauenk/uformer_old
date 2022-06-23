"""

Straightforward testing of a single sample

"""

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#           Imports
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# -- misc --
import os,math,tqdm
import random,pprint
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- optical flow --
import svnlb

# -- network --
import uformer
import uformer

# -- caching results --
import cache_io

# -- cropping --
from torchvision.transforms.functional import center_crop

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#         Helper Funcs
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def get_model(mtype,sigma,ishape,device):
    if mtype == "batched":
        model = uformer.batched.load_model(sigma,uformer_pad=True).to(device)
        model.eval()
    elif mtype == "original":
        model = uformer.refactored.load_model(sigma,"original").to(device)
        model.eval()
    else:
        raise ValueError(f"Uknown mtype [{mtype}]")
    return model

def run_model(mtype,model,sigma,noisy,flows,batch_size,ws,wt):
    with th.no_grad():
        deno = model(noisy,cfg.sigma,flows=flows,
                     ws=cfg.ws,wt=cfg.wt,batch_size=batch_size)
    # with th.no_grad():
    #     if mtype == "batched":
    #         deno = model(noisy,sigma,flows=flows,
    #                      ws=ws,wt=wt,batch_size=batch_size)
    #     elif mtype == "original":
    #         deno = model.run_parts(noisy,sigma,train=False)
    #     else:
    #         raise ValueError(f"Uknown mtype [{mtype}]")
    deno = deno.detach()
    return deno

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#         Main Code Logic
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# -- config --
dname = "set8"
# vid_name = "motorbike"
vid_name = "tractor"
# dname = "div2k"
# vid_name = ""
device = "cuda:0"
sigma = 30.
nframes = -1
ws = 29
wt = 0
# ws = 10
# wt = 10
save_dir = "./output/saved_results/single/"
comp_flow = "false"#"true"
# mtype = "original"
mtype = "batched"
mod_isize = None#256
seed = 123

# -- set seed --
th.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# -- create timer --
timer = uformer.utils.timer.ExpTimer()

# -- data --
cfg = edict({"dname":dname,"sigma":sigma,"ntype":"g"})
data,loaders = data_hub.sets.load(cfg)
if vid_name != "": index = data.te.groups.index(vid_name)
else: index = 0
sample = data.te[index]

# -- unpack --
noisy,clean = sample['noisy'],sample['clean']
noisy,clean = noisy.to(device),clean.to(device)
print("noisy.shape: ",noisy.shape)

# -- handle single image --
if noisy.ndim == 3: # single image
    noisy = noisy[None]
    clean = clean[None]

# -- select frames --
if nframes > 0:
    noisy = noisy[15:nframes+15].contiguous()
    clean = clean[15:nframes+15].contiguous()

# -- crop --
if not(mod_isize is None):
    msize = mod_isize
    noisy = center_crop(noisy,(msize,msize))
    clean = center_crop(clean,(msize,msize))


# -- optical flow --
timer.start("flow")
if comp_flow == "true":
    noisy_np = noisy.cpu().numpy()
    flows = svnlb.compute_flow(noisy_np,cfg.sigma)
    flows = edict({k:th.from_numpy(v).to(device) for k,v in flows.items()})
else:
    flows = None
timer.stop("flow")

# -- network --
model = get_model(mtype,sigma,noisy.shape,device)

# -- test time --
max_npix = 390*390
npix = noisy.shape[0] * np.prod(noisy.shape[-2:])
batch_size = int(max_npix*0.25)
# batch_size = 30000
batch_size = 10000#30000
print(batch_size)
timer.start("deno")
deno = run_model(mtype,model,sigma,noisy,flows,batch_size,ws,wt)
# with th.no_grad():
#     deno = model(noisy,sigma,flows=None,
#                  ws=ws,wt=wt,batch_size=batch_size).detach()
timer.stop("deno")

# -- print psnr --
t = clean.shape[0]
clean_rs = clean.reshape((t,-1))/255.
deno_rs = deno.reshape((t,-1))/255.
psnrs = -10 * th.log10(((clean_rs - deno_rs)**2).mean(1))
print(psnrs)
print(th.mean(psnrs))

# -- save output --
out_dir = Path(save_dir)
noisy_fns = uformer.utils.io.save_burst(noisy/255.,out_dir,"noisy")
deno_fns = uformer.utils.io.save_burst(deno,out_dir,"deno")
print(timer)

