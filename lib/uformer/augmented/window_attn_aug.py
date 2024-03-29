# -- imports --
import math
import torch as th
import torch.nn as nn
from timm.models.layers import trunc_normal_

# -- local imports --
from .embedding_modules import LinearProjection,ConvProjection
from .embedding_modules import LinearProjection_Concat_kv
from .misc_modules import SELayer

# -- imports --
import dnls
from einops import rearrange,repeat
from uformer.utils.misc import assert_nonan,optional,tuple_as_int


########### window-based self-attention #############
class WindowAttentionAugmented(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear',
                 qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.,stride=None,ws=-1,wt=0,k=-1,sb=None,exact=False):

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

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        print("num_heads: ",num_heads)
        if token_projection =='conv':
            self.qkv = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        elif token_projection =='linear_concat':
            self.qkv = LinearProjection_Concat_kv(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # self.se_layer = SELayer(dim) if se_layer else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=0)

    def _qkv_videos(self,x,region,attn_kv,modulator):

        # -- params --
        adj,dil = 0,1
        use_reflect = False
        only_full = False
        device = x.device
        t,h,w,c = x.shape
        vshape = (t,c,h,w)
        stride = self.stride
        border = "zero" if use_reflect else "reflect"
        region = [0,t,0,0,h,w] if region is None else region
        coords = region[2:]
        ps,sb = self.ps,self.sb

        # -- declarations --
        unfold = dnls.iunfold.iUnfold(ps,coords,stride=stride,dilation=dil,
                                      adj=adj,only_full=only_full,border=border)
        qfold = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil,
                                 adj=adj,only_full=only_full,
                                 use_reflect=use_reflect,device=device)
        kfold = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil,
                                 adj=adj,only_full=only_full,
                                 use_reflect=use_reflect,device=device)
        vfold = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil,
                                 adj=adj,only_full=only_full,
                                 use_reflect=use_reflect,device=device)
        wfold = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil,
                                 adj=0,only_full=False,use_reflect=False,
                                 device=device)

        # -- batching info --
        npix = h*w
        coords = region[2:]
        cr_h = coords[2] - coords[0]
        cr_w = coords[3] - coords[1]
        nh = (cr_h-1)//stride+1
        nw = (cr_w-1)//stride+1
        ntotal = t * nh * nw
        div = 2 if npix >= (540 * 960) else 1
        if sb is None: nbatch = ntotal//(t*div)
        else: nbatch = sb
        min_nbatch,max_nbatch = 4096,1024*32
        nbatch = max(nbatch,min_nbatch) # at least "min_batch"
        nbatch = min(nbatch,max_nbatch) # at most "max_batch"
        nbatch = min(nbatch,ntotal) # actualy min is "ntotal"
        nbatches = (ntotal-1) // nbatch + 1

        # -- prepare input vid --
        x = rearrange(x,'t h w c -> t c h w').contiguous()

        # -- for each batch --
        for batch in range(nbatches):

            # -- batch info --
            qindex = min(nbatch * batch,ntotal)
            nbatch_i =  min(nbatch, ntotal - qindex)

            # -- get patches --
            iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch_i,stride,
                                                        region,t,device=device)

            # -- select attention mask --
            attn_kv_i = attn_kv
            if not(attn_kv is None):
                print(attn_kv_i.shape)

            # -- unfold --
            patches = unfold(x,qindex,nbatch_i) # n k pt c ph pw
            patches = rearrange(patches,'n 1 1 c ph pw -> n (ph pw) c')

            # -- optional modulator --
            if not(modulator is None):
                patches += modulator.weight

            # -- transform --
            q, k, v = self.qkv(patches,attn_kv_i)
            q = q * self.scale
            if batch == 0:
                print("q.shape: ",q.shape)

            # -- reshape for folding --
            # num patchees, num heads, (ps ps), num channels
            q = rearrange(q,'n h (ph pw) c -> n 1 1 (h c) ph pw',ph=ps,pw=ps)
            k = rearrange(k,'n h (ph pw) c -> n 1 1 (h c) ph pw',ph=ps,pw=ps)
            v = rearrange(v,'n h (ph pw) c -> n 1 1 (h c) ph pw',ph=ps,pw=ps)

            # -- contiguous --
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            ones = th.ones_like(q)

            # -- folding --
            qfold(q,qindex)
            kfold(k,qindex)
            vfold(v,qindex)
            wfold(ones,qindex)

        # -- folding weights --
        q = qfold.vid / (wfold.vid + 1e-10)
        k = kfold.vid / (wfold.vid + 1e-10)
        v = vfold.vid / (wfold.vid + 1e-10)

        # -- running assert --
        assert_nonan(q)
        assert_nonan(k)
        assert_nonan(v)

        return q,k,v

    def _index_rel_pos(self,rel_pos,qindex,nbatch,nframes):
        if rel_pos is None: return None
        rel_pos_inds = th.arange(qindex,qindex+nbatch) % nframes
        print("rel_pos_inds.shape: ",rel_pos_inds.shape)
        rel_pos_i = rel_pos[rel_pos_inds]
        print("rel_pos_i.shape: ",rel_pos_i.shape)
        return rel_pos_i

    def _index_mask(self,mask,qindex,nbatch,nframes):
        if mask is None: return None
        mask_inds = th.arange(qindex,qindex+nbatch) % nframes
        print("mask_inds.shape: ",mask_inds.shape)
        mask_i = mask[mask_inds]
        print("mask_i.shape: ",mask_i.shape)
        return mask_i

    def _modify_attn(self,attn,rel_pos_i,mask_i):
        # attn.shape = (PS * PS)**2?, nW

        # -- add offset --
        # print("rel_pos.shape: ",rel_pos.shape)
        # rel_pos = rearrange(rel_pos,'nH l c -> nH (l c)')
        # print("rel_pos.shape: ",rel_pos.shape)
        # ratio = attn.size(0)//rel_pos.size(-1)
        # rel_pos = repeat(rel_pos,'nH lc -> (lc d) nH', d = ratio)
        # print("rel_pos.shape: ",rel_pos.shape)
        attn = attn + rel_pos_i
        assert ratio == 1, "What is the ratio?"

        # -- add mask --
        # print("attn.shape: ",attn.shape)
        # print("mask_i.shape: ",mask_i.shape)
        if mask_i is not None:
            atnn += mask_i
        attn = self.softmax(attn)

        # attn = self.attn_drop(attn)
        return attn

    def get_rel_pos(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.win_size[0] * self.win_size[1],
                self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # print(relative_position_bias.shape)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # ratio = attn.size(-1)//relative_position_bias.size(-1)
        # relative_position_bias = repeat(relative_position_bias,
        #                                 'nH l c -> nH l (c d)', d = ratio)
        return relative_position_bias

    def get_flows(self,flows,vshape):
        fflow = optional(flows,'fflow',None)
        bflow = optional(flows,'bflow',None)
        fflow,bflow = None,None
        return fflow,bflow

    def get_xsearch(self,fflow,bflow):
        oh0,ow0,oh1,ow1 = 0,0,0,0
        stride,dil = self.stride,1
        k,ps,ws,wt,pt = self.k,self.ps,self.ws,self.wt,1
        reflect_bounds,use_adj = False,True
        use_k,exact = not(ws == -1),self.exact
        use_search_abs = ws == -1
        xsearch = dnls.search.init("prod",fflow, bflow, k, ps, pt, ws, wt,
                                   oh0, ow0, oh1, ow1, chnls=-1,
                                   dilation=dil, stride=stride,
                                   reflect_bounds=reflect_bounds,
                                   use_k=use_k,use_adj=use_adj,
                                   use_search_abs=use_search_abs,
                                   exact=exact)
        return xsearch

    def get_wsearch(self,fflow,bflow):
        oh0,ow0,oh1,ow1 = 0,0,0,0
        stride,dil = self.stride,1
        k,ps,ws,wt,pt = self.k,self.ps,self.ws,self.wt,1
        reflect_bounds,use_adj = False,True
        use_k,exact = not(ws == -1),self.exact
        use_search_abs = ws == -1
        xsearch = dnls.search.CrossSearchNl(fflow, bflow, k, ps, pt, ws, wt,
                                             oh0, ow0, oh1, ow1, chnls=-1,
                                             dilation=dil, stride=stride,
                                             reflect_bounds=reflect_bounds,
                                             use_k=use_k,use_adj=use_adj,
                                             use_search_abs=use_search_abs,
                                             exact=exact)
        return xsearch

    def get_wpsum(self):
        ps,pt,dil = self.ps,1,1
        reflect_bounds,adj = False,0
        exact = self.exact
        wpsum = dnls.wpsum.WeightedPatchSum(ps, pt, h_off=0, w_off=0, dilation=dil,
                                            reflect_bounds=reflect_bounds,
                                            adj=adj, exact=exact)
        return wpsum

    def forward(self, x, flows, modulator, attn_kv=None, mask=None, region=None):

        # print("start it.")
        # -- init params --
        t,h,w,c = x.shape
        vshape = (t,c,h,w)
        region = [0,t,0,0,h,w] if region is None else region
        stride,sb = self.stride,self.sb
        device,ps = x.device,self.ps
        coords = region[2:]
        adj,dil = 0,1
        only_full = False
        use_reflect = False

        # -- init functions --
        fflow,bflow = self.get_flows(flows,x.shape)
        print("hi.")
        xsearch = self.get_xsearch(fflow,bflow)
        wpsum = self.get_wpsum()
        fold = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil,
                                 adj=adj,only_full=only_full,
                                 use_reflect=use_reflect,device=device)
        wfold = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil,
                                 adj=0,only_full=False,use_reflect=False,
                                 device=device)

        # -- get (q,k,v) videos --
        q_vid,k_vid,v_vid = self._qkv_videos(x,region,attn_kv,modulator)
        rel_pos = self.get_rel_pos()

        # -- batching info --
        npix = h*w
        coords = region[2:]
        cr_h = coords[2] - coords[0]
        cr_w = coords[3] - coords[1]
        nh = (cr_h-1)//stride+1
        nw = (cr_w-1)//stride+1
        ntotal = t * nh * nw
        div = 2 if npix >= (540 * 960) else 1
        if sb is None: nbatch = ntotal//(t*div)
        else: nbatch = sb
        min_nbatch,max_nbatch = 4096,1024*32
        nbatch = max(nbatch,min_nbatch) # at least "min_batch"
        nbatch = min(nbatch,max_nbatch) # at most "max_batch"
        nbatch = min(nbatch,ntotal) # at most "ntotal"
        nbatch = ntotal
        nbatches = (ntotal-1) // nbatch + 1

        # -- iter over batches --
        for batch in range(nbatches):

            # -- batch info --
            qindex = min(nbatch * batch,ntotal)
            nbatch_i =  min(nbatch, ntotal - qindex)

            # -- get patches --
            iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch_i,stride,
                                                        region,t,device=device)
            # -- index mask --
            mask_i = mask#self._index_mask(mask,qindex,nbatch_i,t)

            # -- index relative position offset --
            rel_pos_i = rel_pos#self._index_rel_pos(rel_pos,qindex,nbatch_i,t)

            # -- process each head separately --
            # print("\n"*5)
            # print("-"*30)
            # print(q_vid.shape,self.dim,self.num_heads)
            # print("-"*30)
            # print("\n"*5)
            x,nchnls = [],self.dim//self.num_heads
            for head in range(self.num_heads):

                # -- access channels for head --
                chnls = slice(head * nchnls, (head+1) * nchnls)
                q_vid_head = q_vid[:,chnls].contiguous()
                k_vid_head = k_vid[:,chnls].contiguous()
                v_vid_head = v_vid[:,chnls].contiguous()
                th.cuda.synchronize()

                # -- get attention map --
                print(q_vid_head.shape)
                # print(k_vid_head.shape)
                attn,inds = xsearch(q_vid_head,iqueries,k_vid_head)
                print("attn.shape: ",attn.shape)

                # -- modified --
                # attn = self._modify_attn(attn,rel_pos_i,mask_i)

                # -- compute product --
                x_i = wpsum(v_vid_head,attn,inds)
                x_i = rearrange(x_i,'b c ph pw -> b (ph pw) c')

                # -- append --
                print("x_i.shape: ",x_i.shape)
                x.append(x_i)

            # -- reshape for fold --
            x = th.cat(x,-1)
            print("x.shape: ",x.shape)

            # -- final xform --
            x = self.proj(x)
            # x = self.se_layer(x)
            # x = self.proj_drop(x)
            x = rearrange(x,'b (ph pw) c -> b 1 1 c ph pw ',ph=ps,pw=ps)
            x = x.contiguous()
            ones = th.ones_like(x)

            # -- fold into video --
            fold(x,qindex)
            wfold(ones,qindex)

        # -- fold the video --
        x = fold.vid / (wfold.vid + 1e-10)
        assert_nonan(x)
        # print("though it.")

        return x

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
        if self.token_projection !='linear_concat':
            flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
            #  x = (attn @ v)
            flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)
        else:
            flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N*2
            #  x = (attn @ v)
            flops += nW * self.num_heads * N * N*2 * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += nW * N * self.dim * self.dim
        # print("W-MSA:{%.2f}"%(flops/1e9))
        return flops
