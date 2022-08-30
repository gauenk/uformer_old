# -- imports --
import math
import torch as th
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import repeat

# -- local imports --
from .embedding_modules import LinearProjection,ConvProjection,LinearProjection_Concat_kv
from .embedding_modules import ConvProjectionNoReshape
from .misc_modules import SELayer
from .window_utils import window_partition,window_reverse

# -- project imports --
import dnls
from dnls.utils.inds import get_nums_hw
from einops import rearrange,repeat
from uformer.utils.misc import assert_nonan,optional,tuple_as_int


########### window-based self-attention #############
class NonLocalAttention(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear',
                 qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0., stride=None,ws=-1,wt=0,k=-1,sb=None,exact=False):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # -- set relevant params --
        self.ps = tuple_as_int(win_size)
        self.stride = tuple_as_int(stride)
        self.ws = ws
        self.wt = wt
        self.k = k
        self.sb = sb
        self.exact = exact

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            th.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = th.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = th.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = th.stack(th.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = th.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        if token_projection =='conv':
            self.qkv = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        elif token_projection =='linear_concat':
            self.qkv = LinearProjection_Concat_kv(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        self.qkv_conv = [ConvProjectionNoReshape(dim,num_heads,
                                                 dim//num_heads,
                                                 kernel_size=1,
                                                 bias=qkv_bias,
                                                 q_stride=1,
                                                 v_stride=1,
                                                 k_stride=1)]

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # self.se_layer = SELayer(dim) if se_layer else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def get_qkv_videos(self,vid,attn_kv,modulator):

        # -- params --
        adj,dil = self.ps//2,1
        use_reflect = True
        only_full = True
        device = vid.device
        t,c,h,w = vid.shape
        vshape = (t,c,h,w)
        stride = self.stride
        border = "zero" if use_reflect else "reflect"
        region = None#[0,t,0,0,h,w] if region is None else region
        coords = None
        ps,sb = self.ps,self.sb

        # -- declarations --
        # print("vid.shape: ",vid.shape)
        unfold = dnls.iUnfold(ps,coords,stride=stride,dilation=dil,
                              adj=adj,only_full=only_full,border=border)
        qfold = dnls.iFold(vshape,coords,stride=stride,dilation=dil,
                           adj=adj,only_full=only_full,
                           use_reflect=use_reflect,device=device)
        kfold = dnls.iFold(vshape,coords,stride=stride,dilation=dil,
                           adj=adj,only_full=only_full,
                           use_reflect=use_reflect,device=device)
        vfold = dnls.iFold(vshape,coords,stride=stride,dilation=dil,
                           adj=adj,only_full=only_full,
                           use_reflect=use_reflect,device=device)
        wfold = dnls.iFold(vshape,coords,stride=stride,dilation=dil,
                           adj=adj,only_full=only_full,use_reflect=use_reflect,
                           device=device)

        # -- batching info --
        npix = h*w
        nh,nw = get_nums_hw(vid.shape,stride,ps,dil,False,only_full)
        ntotal = t * nh * nw
        div = 2 if npix >= (540 * 960) else 1
        if sb is None: nbatch = ntotal//(t*div)
        else: nbatch = sb
        min_nbatch,max_nbatch = 4096,1024*32
        nbatch = max(nbatch,min_nbatch) # at least "min_batch"
        nbatch = min(nbatch,max_nbatch) # at most "max_batch"
        nbatch = min(nbatch,ntotal) # actualy min is "ntotal"
        nbatches = (ntotal-1) // nbatch + 1

        # -- for each batch --
        for batch in range(nbatches):

            # -- batch info --
            qindex = min(nbatch * batch,ntotal)
            nbatch_i =  min(nbatch, ntotal - qindex)

            # -- select attention mask --
            attn_kv_i = attn_kv
            # if not(attn_kv is None):
            #     print(attn_kv_i.shape)

            # -- unfold --
            patches = unfold(vid,qindex,nbatch_i) # n k pt c ph pw
            patches = rearrange(patches,'n 1 1 c ph pw -> n (ph pw) c')
            # n,_,_,c,ph,pw = patches.shape
            # patches = patches.view(n,c,ph*pw).permute(0,2,1)

            # -- optional modulator --
            if not(modulator is None):
                patches = patches + modulator.weight

            # -- transform --
            # print("patches.shape: ",patches.shape)
            if not attn_kv_i is None: print("attn_kv_i.shape: ",attn_kv_i.shape)
            q, k, v = self.qkv(patches,attn_kv_i)
            # q = q * self.scale
            # print("q.shape: ",q.shape)
            # if batch == 0:
            #     print("q.shape: ",q.shape)

            # -- reshape for folding --
            # num patchees, num heads, (ps ps), num channels
            q = rearrange(q,'n h (ph pw) c -> n 1 1 (h c) ph pw',ph=ps,pw=ps)
            k = rearrange(k,'n h (ph pw) c -> n 1 1 (h c) ph pw',ph=ps,pw=ps)
            v = rearrange(v,'n h (ph pw) c -> n 1 1 (h c) ph pw',ph=ps,pw=ps)

            # -- contiguous --
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            # ones = th.ones_like(q)

            # -- folding --
            qfold(q,qindex)
            kfold(k,qindex)
            vfold(v,qindex)
            # wfold(ones,qindex)

        # -- folding weights --
        q = qfold.vid# / (wfold.vid + 1e-10)
        k = kfold.vid# / (wfold.vid + 1e-10)
        v = vfold.vid# / (wfold.vid + 1e-10)

        # -- running assert --
        assert_nonan(q)
        assert_nonan(k)
        assert_nonan(v)

        # print("q.shape: ",q.shape)
        # print("k.shape: ",k.shape)
        # print("k.shape: ",v.shape)
        return q,k,v

    def convert_qkv_to_conv(self):

        # -- unpack --
        a,b = self.qkv,self.qkv_conv[0]

        # -- copy q --
        a_params = dict(a.to_q.named_parameters())
        b_params = dict(b.to_q.named_parameters())
        data = a_params['weight'].data[:,:,None,None]
        b_params['weight'].data[...] = data
        data = a_params['bias'].data
        b_params['bias'].data[...] = data

        # -- copy k,v --
        a_params = dict(a.to_kv.named_parameters())
        b_k_params = dict(b.to_k.named_parameters())
        b_v_params = dict(b.to_v.named_parameters())
        data_pair = a_params['weight'].data
        half = data_pair.shape[0]//2
        b_k_params['weight'].data[...] = data_pair[:half,:,None,None]
        b_v_params['weight'].data[...] = data_pair[half:,:,None,None]
        data_pair = a_params['bias'].data
        b_k_params['bias'].data[...] = data_pair[:half]
        b_v_params['bias'].data[...] = data_pair[half:]

    def _index_mask(self,mask,qindex,nbatch,ntotal_t,nframes):
        if mask is None: return None
        mask_inds = th.arange(qindex,qindex+nbatch) % ntotal_t
        mask_i = mask[mask_inds].unsqueeze(1)
        return mask_i

    def _modify_attn(self,attn,rel_pos,mask_i):
        # attn.shape = (PS * PS)**2?, nW

        # -- add offset --
        attn = attn + rel_pos.unsqueeze(0)
        # assert ratio == 1, "What is the ratio?"

        # -- add mask --
        if mask_i is not None:
            attn += mask_i
        attn = self.softmax(attn)
        # attn = self.attn_drop(attn)
        return attn

    def apply_modulator(self,x,modulator,wsize=8):
        # -- if modular weight --
        if not(modulator is None):
            t,c,h,w = x.shape
            mweight = modulator.weight
            nh,nw = h//wsize,w//wsize
            shape_s = '(wh ww) c -> 1 c (rh wh) (rw ww)'
            mweight = repeat(mweight,shape_s,wh=wsize,rh=nh,rw=nw)
            x = x + mweight
        return x

    def get_rel_pos(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.win_size[0] * self.win_size[1],
                self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # print(relative_position_bias.shape)
        relative_position_bias = relative_position_bias.\
            permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # ratio = attn.size(-1)//relative_position_bias.size(-1)
        # relative_position_bias = repeat(relative_position_bias,
        #                                 'nH l c -> nH l (c d)', d = ratio)
        return relative_position_bias

    def get_flows(self,flows,vshape):
        fflow = optional(flows,'fflow',None)
        bflow = optional(flows,'bflow',None)
        fflow,bflow = None,None
        return fflow,bflow

    def silly_fill(self,q_vid,k_vid,v_vid):
        num = int(q_vid.numel())
        anum = th.arange(num)
        q_vid[...] = anum.reshape(q_vid.shape).to(q_vid.device)
        k_vid[...] = anum.reshape(q_vid.shape).to(q_vid.device)
        v_vid[...] = anum.reshape(q_vid.shape).to(q_vid.device)

    def forward(self, vid, modulator, attn_kv=None, mask=None, flows=None):


        # -- prepare input vid --
        vid = rearrange(vid,'t h w c -> t c h w').contiguous()

        # -- modulator --
        # vid = self.apply_modulator(vid,modulator)

        # -- init params --
        t,c,h,w = vid.shape
        vshape = (t,c,h,w)
        stride,sb = self.stride,self.sb
        device,ps = vid.device,self.ps

        # -- params --
        adj,dil = 0,1
        use_reflect = True
        only_full = False
        device = vid.device
        stride = self.stride
        border = "zero" if use_reflect else "reflect"
        sb = self.sb
        pt = 1
        ps,k = self.ps,self.k
        ps_search = 1
        ws,wt = self.ws,self.wt
        ws = self.ps
        # print(ws,wt)
        nheads = self.num_heads

        # -- get flows --
        fflow,bflow = self.get_flows(flows,vid.shape)

        # -- search params --
        oh0, ow0, oh1, ow1 = 0,0,0,0
        use_k = False
        use_adj = False
        # stride0,stride1 = stride,stride
        # print(stride0,stride1,stride)
        use_search_abs = False
        exact = True
        reflect_bounds = use_reflect
        stride = 1

        # -- declarations --
        stride_s = 1
        # unfold = dnls.iUnfold(ps,None,stride=stride,dilation=dil,
        #                       adj=adj,only_full=only_full,border=border)
        # fold = dnls.iFoldz(vshape,None,stride=stride,dilation=dil,
        #                    adj=adj,only_full=only_full,
        #                    use_reflect=use_reflect,device=device)
        search = dnls.search.init("window",fflow, bflow, k,
                                  ps_search, pt, ws, wt, nheads,
                                  chnls=-1,dilation=dil,
                                  stride0=stride_s,stride1=stride_s,
                                  reflect_bounds=reflect_bounds,
                                  use_k=use_k,use_adj=False,full_ws=True,
                                  search_abs=use_search_abs,
                                  h0_off=0,w0_off=0,h1_off=0,w1_off=0,
                                  exact=exact)
        wpsum = dnls.reducers.WeightedPatchSumHeads(ps_search, pt, h_off=0, w_off=0,
                                                    dilation=dil,
                                                    reflect_bounds=reflect_bounds,
                                                    adj=0, exact=exact)
        fold = dnls.iFoldz(vid.shape,None,stride=stride_s,dilation=dil,
                           adj=adj,only_full=only_full,
                           use_reflect=reflect_bounds,device=device)

        # -- batching info --
        npix = h*w
        nh,nw = get_nums_hw(vid.shape,stride_s,ps,dil,False,only_full)
        nh0,nw0 = get_nums_hw(vid.shape,8,ps,dil,False,only_full)
        noriginal = nh0*nw0
        norig = noriginal
        ntotal_t = nh * nw
        ntotal = t * nh * nw
        div = 2 if npix >= (540 * 960) else 1
        if sb is None:
            nbatch = ntotal//(t*div)
            min_nbatch,max_nbatch = 4096,1024*32
            nbatch = max(nbatch,min_nbatch) # at least "min_batch"
            nbatch = min(nbatch,max_nbatch) # at most "max_batch"
        else: nbatch = sb
        nbatch = min(nbatch,ntotal) # actualy min is "ntotal"
        nbatch = ntotal
        nbatches = (ntotal-1) // nbatch + 1

        # -- compute qkv --
        # self.convert_qkv_to_conv()
        # qkv_conv = self.qkv_conv[0].to(vid.device)
        # q_vid, k_vid, v_vid = qkv_conv(vid,attn_kv)
        q_vid, k_vid, v_vid = self.get_qkv_videos(vid,attn_kv,modulator)#attn_kv,modulator)
        # if v_vid.shape[-1] > 9:
        #     print(v_vid[10,::nheads,9,9])

        # -- formatting --
        q_vid = q_vid*self.scale
        rel_pos = self.get_rel_pos()

        # -- view for indexing --
        C = v_vid.shape[1]
        cph = q_vid.shape[1]//self.num_heads

        # -- for each batch --
        for batch in range(nbatches):
            # print("%d/%d" % ((batch+1),nbatches))

            # -- batch info --
            qindex = min(nbatch * batch,ntotal)
            nbatch_i =  min(nbatch, ntotal - qindex)

            # -- compute dists --
            # dists,inds = [],[]
            # for head in range(self.num_heads):
            #     dists_h,inds_h = search(q_vid[head],qindex,nbatch_i,k_vid[head])
            #     # print("dists_h.shape: ",dists_h.shape)
            #     dists.append(dists_h)
            #     inds.append(inds_h)
            # dists = th.stack(dists,1)
            # inds = th.stack(inds,1)
            # print("q_vid.shape: ",q_vid.shape)
            dists,inds = search(q_vid,qindex,nbatch_i,k_vid)
            dists = search.window_softmax(dists,vid.shape)
            # v_vid[...] = 1

            # -- update dists --
            # if not mask is None: print("mask.shape: ",mask.shape)
            # mask_i = None
            # if not mask is None: mask_i = mask.unsqueeze(1)
            # mask_i = self._index_mask(mask,qindex,nbatch_i,ntotal_t,t)
            # if not mask_i is None: print("mask_i.shape:",mask_i.shape)
            # dists = rearrange(dists,'(a b) h c -> a h b c',a=norig)
            # print(dists[0,0,0,0])
            # print("dists.shape: ",dists.shape)
            # dists = self._modify_attn(dists,rel_pos,mask_i)
            # print(dists[0,0,0,0])
            # print("dists.shape: ",dists.shape)
            # print("dists.shape: ",dists.shape)
            # dists = dists[...,None]
            # print("inds.shape: ",inds.shape)

            # -- compute prod --
            # x = []
            # for head in range(self.num_heads):
            #     dists_h = dists[:,head,:,None].contiguous()
            #     inds_h = inds[:,head].contiguous()
            #     x_head = wpsum(v_vid[head],dists_h,inds_h)
            #     # print("x_head.shape: ",x_head.shape)
            #     x.append(x_head)
            # x = th.stack(x,-1)
            # dists = rearrange(dists,'a h b s -> h (a b) s').contiguous()
            # inds = rearrange(inds,'ab h k m -> h ab k m').contiguous()
            # c = q_vid.shape[2]
            # rem = dists.shape[1]//norig
            # print("dists.shape: ",dists.shape,rem,c)
            # x = th.zeros((norig,rem,c*self.num_heads),device=dists.device)
            # print("x.shape: ",x.shape)

            # -- weighted sum --
            x = wpsum(v_vid,dists,inds)
            # patches = rearrange(patches,'b H c 1 1 -> b 1 1 (H c) 1 1')
            # print(norig)
            x = rearrange(x,'(o n) h c 1 1 -> o n (h c)',o=t*norig)
            # print("x.shape: ",x.shape)
            # exit(0)

            # -- final xforms --
            x = self.proj(x)
            # x = self.proj_drop(x)
            # print("x.shape: ",x.shape)

            # -- prepare for folding --
            x = rearrange(x,'o n c -> (o n) 1 1 c 1 1')
            x = x.contiguous()

            # -- folding --
            fold(x,qindex)

        # -- folding weights --
        vid = fold.vid# / (fold.zvid + 1e-10)
        vid = rearrange(vid,'t c h w -> t h w c')

        return vid

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

    def flops(self, H, W):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        N = self.win_size[0]*self.win_size[1]
        nW = H*W/N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(H, W)
        # attn = (q @ k.transpose(-2, -1))
        # x = self.proj(x)
        flops += nW * N * self.dim * self.dim
        # print("W-MSA:{%.2f}"%(flops/1e9))
        return flops
