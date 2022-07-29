

# -- misc --
import os,math,tqdm
import pprint,copy
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import dnls
import data_hub

# -- optical flow --
import svnlb
import skimage

# -- network --
import uformer
import uformer.configs as configs
import uformer.utils.gpu_mem as gpu_mem
from uformer.utils.timer import ExpTimer
from uformer.utils.metrics import compute_psnrs,compute_ssims
from uformer.utils.misc import rslice,write_pickle,read_pickle

# -- generic logging --
import logging
logging.basicConfig()

# -- lightning module --
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only


class UformerLit(pl.LightningModule):

    def __init__(self,batch_size=1,flow=True,
                 isize=None,bw=False,fwd_mode="original",
                 stride=8,ws=29,wt=0,k=100,nversion="noise"):
        super().__init__()
        self.bw = bw
        self.nchnls = 1 if bw else 3
        args = []
        kwargs = dict(fwd_mode=fwd_mode,stride=stride,
                      noise_version=nversion,ws=ws,wt=wt,k=k)
        # self.model = uformer.original.load_model(*args,**kwargs)
        self.model = uformer.augmented.load_model(*args,**kwargs)
        self.batch_size = batch_size
        self.flow = flow
        self.isize = isize
        self.gen_loger = logging.getLogger('lightning')
        self.gen_loger.setLevel("NOTSET")
        self.fwd_mode = fwd_mode
        self.lr = 1e-4

    def forward(self,vid):
        if self.fwd_mode == "dnls_k" or self.fwd_mode == "dnls":
            return self.forward_dnls_k(vid)
        elif self.fwd_mode in ["default","original"]:
            return self.forward_default(vid)
        else:
            msg = f"Uknown ca forward type [{self.fwd_mode}]"
            raise ValueError(msg)

    def forward_dnls_k(self,vid):
        flows = self._get_flow(vid)
        deno = self.model(vid,flows=flows,region=None)
        deno = th.clamp(deno,0.,1.)
        return deno

    def forward_default(self,vid):
        deno = self.model(vid)
        deno = th.clamp(deno,0.,1.)
        return deno

    def _get_flow(self,vid):
        if self.flow == True:
            noisy_np = vid.cpu().numpy()
            if noisy_np.shape[1] == 1:
                noisy_np = np.repeat(noisy_np,3,axis=1)
            sigma_est = skimage.restoration.estimate_sigma(noisy_np,average_sigmas=True,
                                                           multichannel=True,
                                                           channel_axis=1)
            flows = svnlb.compute_flow(noisy_np,sigma_est)
            flows = edict({k:th.from_numpy(v).to(self.device) for k,v in flows.items()})
        else:
            t,c,h,w = vid.shape
            zflows = th.zeros((t,2,h,w)).to(self.device)
            flows = edict()
            flows.fflow,flows.bflow = zflows,zflows
        return flows

    def configure_optimizers(self):
        optim = th.optim.Adam(self.parameters(),lr=self.lr)
        StepLR = th.optim.lr_scheduler.StepLR
        scheduler = StepLR(optim, step_size=5, gamma=0.1)
        return [optim], [scheduler]

    def training_step(self, batch, batch_idx):

        # -- each sample in batch --
        loss = 0 # init @ zero
        nbatch = len(batch['noisy'])
        denos,cleans = [],[]
        for i in range(nbatch):
            deno_i,clean_i,loss_i = self.training_step_i(batch, i)
            loss += loss_i
            denos.append(deno_i)
            cleans.append(clean_i)
        loss = loss / nbatch

        # -- append --
        denos = th.stack(denos)
        cleans = th.stack(cleans)

        # -- log --
        self.log("train_loss", loss.item(), on_step=True,
                 on_epoch=False,batch_size=self.batch_size)

        # -- terminal log --
        psnr = np.mean(compute_psnrs(denos,cleans,div=1.)).item()
        self.gen_loger.info("train_psnr: %2.2f" % psnr)
        # print("train_psnr: %2.2f" % val_psnr)
        self.log("train_loss", loss.item(), on_step=True,
                 on_epoch=False, batch_size=self.batch_size)
        self.log("train_psnr", psnr, on_step=True,
                 on_epoch=False, batch_size=self.batch_size)

        return loss

    def training_step_i(self, batch, i):

        # -- unpack batch
        noisy = batch['noisy'][i]/255.
        clean = batch['clean'][i]/255.

        # -- foward --
        deno = self.forward(noisy)

        # -- report loss --
        loss = th.mean((clean - deno)**2)
        return deno.detach(),clean,loss

    def validation_step(self, batch, batch_idx):

        # -- denoise --
        noisy,clean = batch['noisy'][0]/255.,batch['clean'][0]/255.
        region = batch['region'][0]
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        with th.no_grad():
            deno = self.forward(noisy)
        mem_gb = gpu_mem.print_peak_gpu_stats(False,"val",reset=True)

        # -- loss --
        loss = th.mean((clean - deno)**2)

        # -- report --
        self.log("val_loss", loss.item(), on_step=False,
                 on_epoch=True,batch_size=1)
        self.log("val_gpu_mem", mem_gb, on_step=False,
                 on_epoch=True,batch_size=1)

        # -- terminal log --
        val_psnr = np.mean(compute_psnrs(deno,clean,div=1.)).item()
        self.gen_loger.info("val_psnr: %2.2f" % val_psnr)

    def test_step(self, batch, batch_nb):

        # -- denoise --
        index,region = batch['index'][0],batch['region'][0]
        noisy,clean = batch['noisy'][0]/255.,batch['clean'][0]/255.
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"test",reset=True)
        with th.no_grad():
            deno = self.forward(noisy)
        mem_gb = gpu_mem.print_peak_gpu_stats(False,"test",reset=True)

        # -- compare --
        loss = th.mean((clean - deno)**2)
        psnr = np.mean(compute_psnrs(deno,clean,div=1.)).item()
        ssim = np.mean(compute_ssims(deno,clean,div=1.)).item()

        # -- terminal log --
        self.log("psnr", psnr, on_step=True, on_epoch=False, batch_size=1)
        self.log("ssim", ssim, on_step=True, on_epoch=False, batch_size=1)
        self.log("index",  int(index.item()),on_step=True,on_epoch=False,batch_size=1)
        self.log("mem_gb",  mem_gb, on_step=True, on_epoch=False, batch_size=1)
        self.gen_loger.info("te_psnr: %2.2f" % psnr)

        # -- log --
        results = edict()
        results.test_loss = loss.item()
        results.test_psnr = psnr
        results.test_ssim = ssim
        results.test_gpu_mem = mem_gb
        results.test_index = index.cpu().numpy().item()
        return results

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def _accumulate_results(self,each_me):
        for key,val in each_me.items():
            if not(key in self.metrics):
                self.metrics[key] = []
            if hasattr(val,"ndim"):
                ndim = val.ndim
                val = val.cpu().numpy().item()
            self.metrics[key].append(val)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        print("logging metrics: ",metrics,step)

    def on_train_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_train_batch_end(self, trainer, pl_module, outs,
                           batch, batch_idx, dl_idx):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)


    def on_validation_batch_end(self, trainer, pl_module, outs,
                                batch, batch_idx, dl_idx):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_batch_end(self, trainer, pl_module, outs,
                          batch, batch_idx, dl_idx):
        self._accumulate_results(outs)



def remove_lightning_load_state(state):
    names = list(state.keys())
    for name in names:
        name_new = name.split(".")[1:]
        name_new = ".".join(name_new)
        state[name_new] = state[name]
        del state[name]
