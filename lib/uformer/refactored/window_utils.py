
# -- linalg --
import torch as th
from einops import rearrange,repeat

# -- fold/unfold --
import torch.nn.functional as F

# -- proposed fold/unfold --
import dnls


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#                      Window Partition
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def window_partition(x, win_size, dilation_rate=1, coords=None):
    if coords is None:
        return impl_window_partition_coords(x, win_size, dilation_rate, coords)
        # return impl_window_partition(x, win_size, dilation_rate=1)
    else:
        return impl_window_partition_coords(x, win_size, dilation_rate, coords)

def impl_window_partition_coords(x, win_size, dilation_rate=1, coords=None):
    unfolder = dnls.iunfold.iUnfold(win_size,coords,stride=win_size,dilation=1,adj=True)

    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        # print("[unfolding]: x.shape: ",x.shape)
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
        exit(0)
    else:
        # print("[dil==1] [unfolding]: x.shape: ",x.shape)

        # -- ours --
        x_ours = rearrange(x,'t h w c -> t c h w').contiguous()
        ps = win_size
        stride = win_size
        dil = dilation_rate
        n_h = (H - (ps-1)*dil - 1)//stride + 1
        n_w = (W - (ps-1)*dil - 1)//stride + 1
        n_total = B * n_h * n_w
        x_ours = unfolder(x_ours,0,n_total)
        shape_str = "n 1 1 c ph pw -> n ph pw c"
        x_ours = rearrange(x_ours,shape_str).contiguous()

    return x_ours

def impl_window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#                      Window Reverse
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def window_reverse(windows, win_size, H, W, dilation_rate=1, coords=None):
    if coords is None:
        return impl_window_reverse_coords(windows, win_size, H, W, dilation_rate)
        # return impl_window_reverse(windows, win_size, H, W, dilation_rate)
    else:
        return impl_window_reverse_coords(windows, win_size, H, W, dilation_rate,coords)

def impl_window_reverse_coords(windows, win_size, H, W, dilation_rate=1, coords=None):
    # B' ,Wh ,Ww ,C
    _,wh,ww,C = windows.shape
    T = 32
    folder = dnls.ifold.iFold((T,C,H,W),coords,stride=win_size,dilation=1,adj=True)
    wfolder = dnls.ifold.iFold((T,C,H,W),coords,stride=win_size,dilation=1,adj=True)

    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        # print("[dil!=1] [folding]: x.shape: ",x.shape)
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x_ours = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate,
                        padding=4*(dilation_rate-1),stride=win_size)
    else:
        x_ours = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
        shape_str = "t h w ph pw c -> (t h w) 1 1 c ph pw"
        x_ours = rearrange(x_ours,shape_str).contiguous()
        x_ours = folder(x_ours,0)
        x_ours = rearrange(x_ours,'t c h w -> t h w c')

    return x_ours

def impl_window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
