
# -- misc --
import os,math,tqdm
import pprint
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

# -- caching results --
import cache_io

# -- network --
import uformer

def run_exp(cfg):

    # -- set device --
    th.cuda.set_device(int(cfg.device.split(":")[1]))

    # -- init results --
    results = edict()
    results.psnrs = []
    results.adapt_psnrs = []
    results.deno_fns = []
    results.vid_frames = []
    results.vid_name = []
    results.timer_flow = []
    results.timer_adapt = []
    results.timer_deno = []

    # -- network --
    model = uformer.batched.load_model(cfg.sigma).to(cfg.device)
    model.eval()

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    data,loaders = data_hub.sets.load(cfg)
    groups = data.te.groups
    indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]
    def fbnds(fnums,lb,ub): return (lb <= np.min(fnums)) and (ub >= np.max(fnums))
    indices = [i for i in indices if fbnds(data.te.paths['fnums'][groups[i]],
                                           cfg.frame_start,cfg.frame_end)]
    for index in indices:

        # -- clean memory --
        th.cuda.empty_cache()

        # -- unpack --
        sample = data.te[index]
        noisy,clean = sample['noisy'],sample['clean']
        noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
        vid_frames = sample['fnums']
        # print("[%d] noisy.shape: " % index,noisy.shape)

        # -- create timer --
        timer = uformer.utils.timer.ExpTimer()

        # -- size --
        nframes = noisy.shape[0]
        ngroups = int(25 * 37./nframes)
        batch_size = 390*39#ngroups*1024

        # -- optical flow --
        timer.start("flow")
        if cfg.flow == "true":
            noisy_np = noisy.cpu().numpy()
            flows = svnlb.compute_flow(noisy_np,cfg.sigma)
            flows = edict({k:th.from_numpy(v).to(cfg.device) for k,v in flows.items()})
        else:
            flows = None
        timer.stop("flow")

        # -- internal adaptation --
        timer.start("adapt")
        run_internal_adapt = cfg.internal_adapt_nsteps > 0
        run_internal_adapt = run_internal_adapt and (cfg.internal_adapt_nepochs > 0)
        adapt_psnrs = [0.]
        if run_internal_adapt:
            adapt_psnrs = model.run_internal_adapt(noisy,cfg.sigma,flows=flows,
                          ws=cfg.ws,wt=cfg.wt,batch_size=batch_size,
                          nsteps=cfg.internal_adapt_nsteps,
                          nepochs=cfg.internal_adapt_nepochs,
                          sample_mtype=cfg.adapt_mtype,
                          clean_gt = clean,
                          region_gt = [2,4,128,256,256,384]
            )
        timer.stop("adapt")

        # -- denoise --
        batch_size = 390*100
        timer.start("deno")
        with th.no_grad():
            deno = model(noisy,cfg.sigma,flows=flows,
                         ws=cfg.ws,wt=cfg.wt,batch_size=batch_size)
        timer.stop("deno")

        # -- save example --
        out_dir = Path(cfg.saved_dir) / str(cfg.uuid)
        deno_fns = uformer.utils.io.save_burst(deno,out_dir,"deno")

        # -- psnr --
        t = clean.shape[0]
        deno = deno.detach()
        clean_rs = clean.reshape((t,-1))/255.
        deno_rs = deno.reshape((t,-1))/255.
        mse = th.mean((clean_rs - deno_rs)**2,1)
        psnrs = -10. * th.log10(mse).detach()
        psnrs = list(psnrs.cpu().numpy())
        print(psnrs)

        # -- append results --
        results.psnrs.append(psnrs)
        results.adapt_psnrs.append(adapt_psnrs)
        results.deno_fns.append(deno_fns)
        results.vid_frames.append(vid_frames)
        results.vid_name.append([cfg.vid_name])
        for name,time in timer.items():
            results[name].append(time)

    return results


def default_cfg():
    # -- config --
    cfg = edict()
    cfg.nframes = 5
    cfg.frame_start = 0
    cfg.frame_end = 4
    cfg.saved_dir = "./output/saved_results/"
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/uformer/output/checkpoints/"
    cfg.isize = None
    cfg.num_workers = 1
    cfg.device = "cuda:0"
    # cfg.isize = "512_512"
    # cfg.isize = "256_256"
    # cfg.isize = "128_128"
    return cfg

def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "test_rgb_net" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- get mesh --
    # dnames = ["toy"]
    # vid_names = ["text_tourbus"]
    # mtypes = ["rand"]
    mtypes = ["rand","sobel"]
    dnames = ["set8"]
    vid_names = ["snowboard","sunflower","tractor","motorbike",
                 "hypersmooth","park_joy","rafting","touchdown"]
    # vid_names = ["tractor"]
    sigmas = [30,50]#,50]
    internal_adapt_nsteps = [300]
    internal_adapt_nepochs = [5]
    # ws,wt = [29],[0]
    ws,wt = [7],[10]
    flow = ["true"]
    isizes = [None,"512_512","256_256"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "flow":flow,"ws":ws,"wt":wt,"adapt_mtype":mtypes,"isize":isizes}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh
    # ws,wt = [29],[0]
    # flow = ["false"]
    # exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
    #              "internal_adapt_nsteps":internal_adapt_nsteps,
    #              "internal_adapt_nepochs":internal_adapt_nepochs,
    #              "flow":flow,"ws":ws,"wt":wt,"adapt_mtype":mtypes}
    # exps += cache_io.mesh_pydicts(exp_lists) # create mesh

    # pp.pprint(exps)
    # for exp in exps:
    #     if exp.internal_adapt_nsteps == 0:
    #         exp.internal_adapt_nepochs = 2

    # -- group with default --
    cfg = default_cfg()
    cache_io.append_configs(exps,cfg) # merge the two

    # -- run exps --
    nexps = len(exps)
    for exp_num,exp in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- logic --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        # cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    print(records)
    print(records.filter(like="timer"))

    # -- print by dname,sigma --
    for dname,ddf in records.groupby("dname"):
        # field = "internal_adapt_nsteps"
        field = "adapt_mtype"
        for adapt,adf in ddf.groupby(field):
            adapt_psnrs = np.stack(adf['adapt_psnrs'].to_numpy())
            print("adapt_psnrs.shape: ",adapt_psnrs.shape)
            print(adapt_psnrs)
            for cflow,fdf in adf.groupby("flow"):
                for ws,wsdf in fdf.groupby("ws"):
                    for wt,wtdf in wsdf.groupby("wt"):
                        print("adapt,ws,wt,cflow: ",adapt,ws,wt,cflow)
                        for sigma,sdf in wtdf.groupby("sigma"):
                            ave_psnr,ave_time,num_vids = 0,0,0
                            for vname,vdf in sdf.groupby("vid_name"):
                                print("vdf.psnrs.shape: ",vdf.psnrs.shape)
                                ave_psnr += vdf.psnrs[0].mean()
                                ave_time += vdf['timer_deno'].iloc[0]/len(vdf)
                                num_vids += 1
                            ave_psnr /= num_vids
                            ave_time /= num_vids
                            total_frames = len(sdf)
                            fields = (sigma,ave_psnr,ave_time,total_frames)
                            print("[%d]: %2.3f @ ave %2.2f sec for %d frames" % fields)


if __name__ == "__main__":
    main()

# 29.768 @ ?
# 30.047 @ 1297.05/497
# ...
