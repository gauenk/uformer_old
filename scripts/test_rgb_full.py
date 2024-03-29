
# -- misc --
import os,math,tqdm,h5py
import hdf5storage
from skimage.metrics import structural_similarity as compute_ssim_ski
import pprint
pp = pprint.PrettyPrinter(indent=4)

# -- vision --
import scipy.io
from PIL import Image

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
from uformer.utils.misc import optional,rslice_pair,assert_nonan
from uformer.utils.misc import stacked2full,full2blocks
from uformer.utils.misc import tiles2img,img2tiles

def run_exp(cfg):

    # -- set device --
    th.cuda.set_device(int(cfg.device.split(":")[1]))

    # -- init results --
    results = edict()
    results.psnrs = []
    results.ssims = []
    results.noisy_psnrs = []
    results.noisy_ssims = []
    results.adapt_psnrs = []
    results.deno_fns = []
    results.vid_frames = []
    results.vid_name = []
    results.timer_flow = []
    results.timer_adapt = []
    results.timer_deno = []

    # -- network --
    if cfg.model_type == "original":
        model = uformer.original.load_model().to(cfg.device)
        tile_h,tile_w = 256,256
        # tile_h,tile_w = 1024,1024
    elif cfg.model_type == "aug_original":
        fwd_mode = "original"
        model = uformer.augmented.load_model(cfg.sigma,fwd_mode=fwd_mode,
                                             stride=cfg.stride,sb=32*1024)
        tile_h,tile_w = 2048,2048
    elif cfg.model_type == "aug_dnls_k":
        fwd_mode = "dnls_k"
        model = uformer.augmented.load_model(cfg.sigma,fwd_mode=fwd_mode,
                                             stride=cfg.stride)
        tile_h,tile_w = 2048,2048
    elif cfg.model_type == "none":
        model = None
        tile_h,tile_w = 2048,2048
    else:
        raise ValueError(f"Uknown model_type [{model_type}]")
    if not(model is None):
        model.eval()
    imax = 255.

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    groups = data[cfg.dset].groups
    indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]

    # -- optional filter --
    frame_start = optional(cfg,"frame_start",0)
    frame_end = optional(cfg,"frame_end",0)
    if frame_start >= 0 and frame_end > 0:
        def fbnds(fnums,lb,ub): return (lb <= np.min(fnums)) and (ub >= np.max(fnums))
        indices = [i for i in indices if fbnds(data[cfg.dset].paths['fnums'][groups[i]],
                                               cfg.frame_start,cfg.frame_end)]

    # -- each subsequence with video name --
    for index in indices:

        # -- clean memory --
        th.cuda.empty_cache()

        # -- unpack --
        sample = data.val[index]
        noisy,clean = sample['noisy'],sample['clean']
        noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
        vid_frames,region = sample['fnums'],optional(sample,'region',None)
        blocks = data[cfg.dset].blocks
        # print(blocks.T,blocks.shape)
        # t,l = blocks[:,0].min(),blocks[:,1].min()
        # b,r = blocks[:,1].max()+256,blocks[:,1].max()+256
        fstart = min(vid_frames)
        # noisy,clean = rslice_pair(noisy,clean,region)
        print("[%d] noisy.shape: " % index,noisy.shape)
        # noisy = rearrange(noisy,'1 c (rh h) (rw w) -> (rh rw) c h w',rh=2,rw=2)
        # clean = rearrange(clean,'1 c (rh h) (rw w) -> (rh rw) c h w',rh=2,rw=2)
        print("[%d] noisy.shape: " % index,noisy.shape)
        t,c,H,W = noisy.shape
        assert_nonan(noisy)
        noisy,nh,nw = img2tiles(noisy,tile_h,tile_w)
        print("noisy.shape: ",noisy.shape)

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
            if model is None:
                deno = noisy
            else:
                t = noisy.shape[0]
                deno = []
                for ti in range(t):
                    deno_i = model(noisy[[ti]]/imax)
                    # deno_i = noisy[[ti]]/imax
                    deno.append(deno_i)
                deno = th.cat(deno)
                deno = th.clamp(deno,0.,1.)*imax
        timer.stop("deno")

        # -- create deno img from tiles --
        noisy = tiles2img(noisy,nh,nw,H,W)
        deno = tiles2img(deno,nh,nw,H,W)

        #
        # -- save example --
        #

        out_dir = Path(cfg.saved_dir) / cfg.dname / cfg.vid_name / cfg.model_type

        # -- full output image --
        out_dir_full = out_dir / "full"
        uformer.utils.io.save_burst(deno,out_dir,"deno",
                                    fstart=fstart,div=1.,fmt="np")
        uformer.utils.io.save_burst(deno,out_dir,"deno",
                                    fstart=fstart,div=1.,fmt="png")
        # -- output blocks  --
        deno = full2blocks(deno,blocks)
        clean = full2blocks(clean,blocks)
        noisy = full2blocks(noisy,blocks)
        out_dir_blocks = out_dir / "blocks"
        deno_fns = uformer.utils.io.save_burst(deno,out_dir,"deno",
                                               fstart=fstart,div=1.,fmt="np")
        uformer.utils.io.save_burst(deno,out_dir,"deno",
                                    fstart=fstart,div=1.,fmt="image")


        # -- psnr --
        noisy_psnrs = compute_psnr(clean,noisy,div=imax)
        psnrs = compute_psnr(clean,deno,div=imax)
        noisy_ssims = compute_ssim(clean,noisy,div=imax)
        ssims = compute_ssim(clean,deno,div=imax)

        # -- append results --
        results.noisy_psnrs.append(noisy_psnrs)
        results.psnrs.append(psnrs)
        results.noisy_ssims.append(noisy_ssims)
        results.ssims.append(ssims)
        results.adapt_psnrs.append(adapt_psnrs)
        results.deno_fns.append(deno_fns)
        results.vid_frames.append(vid_frames)
        results.vid_name.append([cfg.vid_name])
        for name,time in timer.items():
            results[name].append(time)

    return results

def compute_ssim(clean,deno,div=255.):
    nframes = clean.shape[0]
    ssims = []
    for t in range(nframes):
        clean_t = clean[t].cpu().numpy().squeeze().transpose((1,2,0))/div
        deno_t = deno[t].cpu().numpy().squeeze().transpose((1,2,0))/div
        ssim_t = compute_ssim_ski(clean_t,deno_t,channel_axis=-1)
        ssims.append(ssim_t)
    ssims = np.array(ssims)
    return ssims

def prepare_sidd(records,name):

    # -- load denoised files --
    deno_all_fns = list(np.stack(records['deno_fns'].to_numpy()))
    denos = []
    for deno_vid_fns in deno_all_fns:
        deno_vid = []
        for fn_t in deno_vid_fns[0]:
            if "png" in fn_t:
                frame_t = np.array(Image.open(fn_t))
            else:
                frame_t = np.load(fn_t).transpose(1,2,0)
            deno_vid.append(frame_t)
        deno_vid = np.stack(deno_vid)
        denos.append(deno_vid)
    denos = np.stack(denos)
    ntotal = np.prod(denos.shape)

    # -- average time --
    times = list(np.stack(records['timer_deno'].to_numpy()))
    times_mp = []
    for vid_times in times:
        vid_time = np.mean(vid_times)
        times_mp.append(vid_time)
    time_mp = np.mean(times_mp)
    time_mp = time_mp * 1024 * 1024 / ntotal
    # print(time_mp)
    # print(denos.shape)

    # -- out dir --
    out_dir = Path("output/sidd_submit/%s/" % name)
    if not out_dir.exists():
        out_dir.mkdir()

    # -- filenames --
    fn_og = "output/sidd_submit/SubmitSrgbFromMatlab.mat"
    fn = out_dir / "SubmitSrgb.mat"
    if fn.exists():
        os.remove(str(fn)) # remove old
    fn = str(fn)

    # -- read --
    data = hdf5storage.loadmat(fn_og)
    print(data['DenoisedBlocksSrgb'].shape)
    del data['DenoisedBlocksSrgb']
    data['DenoisedBlocksSrgb'] = denos
    print(data['DenoisedBlocksSrgb'].shape)
    hdf5storage.savemat(fn,data)

def compute_psnr(clean,deno,div=255.):
    t = clean.shape[0]
    deno = deno.detach()
    clean_rs = clean.reshape((t,-1))/div
    deno_rs = deno.reshape((t,-1))/div
    mse = th.mean((clean_rs - deno_rs)**2,1)
    psnrs = -10. * th.log10(mse).detach()
    psnrs = psnrs.cpu().numpy()
    return psnrs

def default_cfg():
    # -- config --
    cfg = edict()
    cfg.nframes = 0
    cfg.frame_start = 0
    cfg.frame_end = 0
    cfg.saved_dir = "./output/saved_results/"
    cfg.num_workers = 1
    cfg.device = "cuda:0"
    cfg.sigma = 50. # use large sigma to approx real noise for optical flow
    return cfg

def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    # print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "sidd_bench_full"
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- select data --
    dnames = ["sidd_bench_full"]
    dset = ["val"]
    vid_names = ["%02d" % x for x in np.arange(0,40)]
    # vid_names = [vid_names[0]]
    # # vid_names = vid_names[35:]

    # -- get mesh --
    internal_adapt_nsteps = [300]
    internal_adapt_nepochs = [0]
    flow = ["false"]
    ws,wt = [7],[10]
    mtypes = ["rand"]
    isizes = ["none"]
    stride = [8]
    model_type = ["aug_original"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"dset":dset,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "flow":flow,"ws":ws,"wt":wt,"model_type":model_type,
                 "adapt_mtype":mtypes,"isize":isizes,"stride":stride}
    exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh

    # -- exps version 2 --
    exp_lists['dnames'] = ["sidd_bench"]
    exp_lists['stride'] = [8]
    exp_lists['model_type'] = ['original']
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    exps = exps_a + exps_b

    # -- group with default --
    cfg = default_cfg()
    # cfg.nframes = 4
    # cfg.frame_start = 0
    # cfg.frame_end = cfg.nframes-1
    # cfg.isize = "256_256"
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
        if exp.model_type == "none":
            cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)

    for model_type,mdf in records.groupby("model_type"):
        for stride,sdf in mdf.groupby("stride"):
            ssims_m,psnrs_m = [],[]
            for vid_name,vdf in sdf.groupby("vid_name"):
                ssims = np.stack(np.array(vdf['ssims']))
                psnrs = np.stack(np.array(vdf['psnrs']))
                ssims_m.append(ssims.mean())
                psnrs_m.append(psnrs.mean())
            ssims_m = np.mean(ssims_m)
            psnrs_m = np.mean(psnrs_m)
            print(model_type,stride,psnrs_m,ssims_m)

    for model_type,mdf in records.groupby('model_type'):
        print(mdf['deno_fns'])
        prepare_sidd(mdf,model_type)
    exit(0)
    print(records)
    print(records.filter(like="timer"))
    print(records['psnrs'].mean())
    ssims = np.stack(np.array(records['ssims']))
    psnrs = np.stack(np.array(records['psnrs']))
    print(ssims)
    print(psnrs)
    print(psnrs.shape)
    print(psnrs.mean())
    print(ssims.mean())
    exit(0)

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
