# -- imports --
import math
import torch as th
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import repeat

# -- local imports --
from .embedding_modules import LinearProjection,ConvProjection,LinearProjection_Concat_kv
from .misc_modules import SELayer

# -- project imports --
import dnls
from einops import rearrange,repeat
from uformer.utils.misc import assert_nonan,optional,tuple_as_int


########### window-based self-attention #############
class WindowAttention(nn.Module):
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

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
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
        self.softmax = nn.Softmax(dim=-1)


    def _index_mask(self,mask,qindex,nbatch,ntotal_t,nframes):
        if mask is None: return None
        mask_inds = th.arange(qindex,qindex+nbatch) % ntotal_t
        # print(mask_inds.min(),mask_inds.max())
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
        attn = self.attn_drop(attn)
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

    def forward(self, vid, modulator, attn_kv=None, mask=None):
        # -- init params --
        t,h,w,c = vid.shape
        vshape = (t,c,h,w)
        region = None
        region = [0,t,0,0,h,w] if region is None else region
        stride,sb = self.stride,self.sb
        device,ps = vid.device,self.ps
        coords = region[2:]

        # -- params --
        adj,dil = ps//2,1
        use_reflect = True
        only_full = True
        device = vid.device
        t,h,w,c = vid.shape
        vshape = (t,c,h,w)
        stride = self.stride
        border = "zero" if use_reflect else "reflect"
        region = [0,t,0,0,h,w] if region is None else region
        coords = region[2:]
        ps,sb = self.ps,self.sb

        # -- declarations --
        unfold = dnls.iunfold.iUnfold(ps,coords,stride=stride,dilation=dil,
                                      adj=adj,only_full=only_full,border=border)
        fold = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil,
                                adj=adj,only_full=only_full,
                                use_reflect=use_reflect,device=device)
        wfold = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil,
                                 adj=adj,only_full=only_full,use_reflect=use_reflect,
                                 device=device)

        # -- batching info --
        npix = h*w
        coords = region[2:]
        cr_h = coords[2] - coords[0]
        cr_w = coords[3] - coords[1]
        if only_full:
            nh = (cr_h-dil*(ps-1)-1)//stride+1
            nw = (cr_w-dil*(ps-1)-1)//stride+1
        else:
            nh = (cr_h-1)//stride+1
            nw = (cr_w-1)//stride+1
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
        # print(nbatch,sb)
        nbatches = (ntotal-1) // nbatch + 1

        # -- prepare input vid --
        vid = rearrange(vid,'t h w c -> t c h w').contiguous()

        # -- misc offsets --
        rel_pos = self.get_rel_pos()

        # -- for each batch --
        for batch in range(nbatches):
            # print("%d/%d" % ((batch+1),nbatches))

            # -- batch info --
            qindex = min(nbatch * batch,ntotal)
            nbatch_i =  min(nbatch, ntotal - qindex)

            # -- get patches --
            # iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch_i,stride,
            #                                             region,t,device=device)

            # -- select attention mask --
            attn_kv_i = attn_kv
            if not(attn_kv is None):
                print("[aug.wattn] attn_kv_i.shape: ",attn_kv_i.shape)
                exit(0)

            # -- unfold --
            patches = unfold(vid,qindex,nbatch_i) # n k pt c ph pw
            patches = rearrange(patches,'n 1 1 c ph pw -> n (ph pw) c')
            B_, N, C = patches.shape

            # -- optional modulator --
            if not(modulator is None):
                patches = patches + modulator.weight

            # -- transform --
            q, k, v = self.qkv(patches,attn_kv_i)
            q = q * self.scale

            # -- compute attn --
            attn = (q @ k.transpose(-2, -1))
            # if not(mask is None): print("[aug.wattn] mask.shape:",mask.shape,ntotal_t)
            mask_i = self._index_mask(mask,qindex,nbatch_i,ntotal_t,t)
            attn = self._modify_attn(attn,rel_pos,mask_i)

            # -- compute weighted sum --
            x = (attn @ v)
            x = x.transpose(1, 2).reshape(B_, N, C)

            # -- final xforms --
            x = self.proj(x)
            # x = self.se_layer(x)
            x = self.proj_drop(x)

            # -- prepare for folding --
            x = rearrange(x,'n (ph pw) c -> n 1 1 c ph pw',ph=ps,pw=ps)
            x = x.contiguous()
            ones = th.ones_like(x)

            # -- folding --
            fold(x,qindex)
            wfold(ones,qindex)

        # -- folding weights --
        vid = fold.vid / (wfold.vid + 1e-10)
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
