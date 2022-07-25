"""
A primary block of Uformer

"""

# -- misc --
import math

# -- timm modules --
from timm.models.layers import DropPath, to_2tuple

# -- torch --
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# -- local imports --
from .window_utils import window_partition,window_reverse
from .window_attn import WindowAttention
from .window_attn_aug import WindowAttentionAugmented
from .embedding_modules import Mlp,LeFF

class BasicUformerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear',token_mlp='ffn',se_layer=False,
                 fwd_mode="dnls_k",stride=None,ws=-1,wt=0,k=-1,sb=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint


        # -- relevant params --
        self.fwd_mode = fwd_mode
        self.stride = stride
        self.ws = ws
        self.wt = wt
        self.k = k
        self.sb = sb

        # build blocks
        is_list = isinstance(drop_path, list)
        block = LeWinTransformerBlockRefactored
        self.blocks = nn.ModuleList([
            block(dim=dim, input_resolution=input_resolution,
                  num_heads=num_heads, win_size=win_size,
                  shift_size=0 if (i % 2 == 0) else win_size // 2,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop, attn_drop=attn_drop,
                  drop_path=drop_path[i] if is_list else drop_path,
                  norm_layer=norm_layer,
                  token_projection=token_projection,
                  token_mlp=token_mlp,se_layer=se_layer,
                  fwd_mode=fwd_mode,stride=stride,
                  ws=ws,wt=wt,k=k,sb=sb)
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x, flows=None, mask=None, region=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, flows, region)
            else:
                x = blk(x,flows,mask,region)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops

class LeWinTransformerBlockRefactored(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 token_projection='linear',token_mlp='leff',se_layer=False,
                 fwd_mode="dnls_k",stride=None,ws=-1,wt=0,k=-1,sb=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp

        # -- relevant params --
        self.fwd_mode = fwd_mode
        self.stride = stride
        self.ws = ws
        self.wt = wt
        self.k = k
        self.sb = sb

        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        if fwd_mode == "dnls_k":
            self.attn = WindowAttentionAugmented(
                dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop,token_projection=token_projection,se_layer=se_layer,
                stride=stride,ws=ws,wt=wt,k=k,sb=sb)
        elif fwd_mode == "original":
            self.attn = WindowAttention(
                dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop,token_projection=token_projection,se_layer=se_layer,
                stride=stride,ws=ws,wt=wt,k=k,sb=sb)
        else:
            raise ValueError(f"Uknown fwd_mode [{fwd_mode}]")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp=='ffn':
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                           act_layer=act_layer, drop=drop)
        else:
            self.mlp = LeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x, flows=None, mask=None, region=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H,W)).permute(0,2,3,1)
            input_mask_windows = window_partition(input_mask, self.win_size, region=region) # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2)*attn_mask.unsqueeze(1) # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = th.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            # nW, win_size, win_size, 1
            shift_mask_windows = window_partition(shift_mask, self.win_size,
                                                  stride=self.stride, region=region)
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask

            # print("attn_mask.shape: ",attn_mask.shape)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # print("[ref]: ",x.shape)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = th.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # -- window region with channel attention --
        stride = self.stride
        shifted_x = self._window_region(shifted_x,attn_mask,stride,flows,region,H,W,C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = th.roll(shifted_x, shifts=(self.shift_size, self.shift_size),
                        dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        # print("x.shape: ",x.shape)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask
        return x

    def _window_region(self,shifted_x,attn_mask,stride,flows,region,H,W,C):
        if self.fwd_mode == "dnls_k":
            return self._window_region_dnls_k(shifted_x,attn_mask,stride,
                                              flows,region,H,W,C)
        elif self.fwd_mode == "original":
            return self._window_region_original(shifted_x,attn_mask,stride,region,H,W,C)
        else:
            raise ValueError(f"Uknown forward mode [{self.fwd_mode}]")

    def _window_region_dnls_k(self,shifted_x,attn_mask,stride,flows,region,H,W,C):

        # partition windows
        # x_windows = window_partition(shifted_x, self.win_size,
        #                              stride=stride, region=region)  # nW*B, win_size, win_size, C  N*C->C
        # x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        # print("x_windows.shape: ",x_windows.shape)

        # W-MSA/SW-MSA
        # print("pre-attn.")
        attn_windows = self.attn(shifted_x, flows, mask=attn_mask)
        # nW*B, win_size*win_size, C
        # print("attn_windows.shape: ",attn_windows.shape)
        # print("post-attn.")

        # merge windows
        # attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        # shifted_x = window_reverse(attn_windows, self.win_size, H, W,
        #                            stride=stride, region=region)  # B H' W' C
        return attn_windows

    def _window_region_original(self,shifted_x,attn_mask,stride,region,H,W,C):

        # partition windows
        # x_windows = window_partition(shifted_x, self.win_size,
        #                              stride=stride, region=region)  # nW*B, win_size, win_size, C  N*C->C
        # x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        # print("x_windows.shape: ",x_windows.shape)

        # W-MSA/SW-MSA
        # print("pre-attn.")
        attn_windows = self.attn(shifted_x, mask=attn_mask)  # nW*B, win_size*win_size, C
        # print("attn_windows.shape: ",attn_windows.shape)
        # print("post-attn.")

        # merge windows
        # attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        # shifted_x = window_reverse(attn_windows, self.win_size, H, W,
        #                            stride=stride, region=region)  # B H' W' C
        return attn_windows

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H,W)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops

