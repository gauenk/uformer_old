
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

def window_partition(x, win_size, dilation_rate=1, stride=None, region=None):
    if region is None:
        return impl_window_partition_region(x, win_size, dilation_rate, stride, region)
        # return impl_window_partition(x, win_size, dilation_rate=1)
    else:
        return impl_window_partition_region(x, win_size, dilation_rate, stride, region)

def impl_window_partition_region(x, win_size, dilation_rate=1, stride=None, region=None):
    stride = win_size if stride is None else stride
    coords = None if region is None else region[2:]
    adj = win_size//2
    only_full = True
    unfolder = dnls.iunfold.iUnfold(win_size,coords,stride=stride,
                                    dilation=1,adj=adj,only_full=only_full)

    B, H, W, C = x.shape

    # -- compute shapes --
    dil = dilation_rate
    ps = win_size
    B, H, W, C = x.shape
    nH = (H - (ps-1)*dil - 1)//stride + 1
    nW = (W - (ps-1)*dil - 1)//stride + 1

    if dilation_rate !=1 :
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=stride) # B, C*Wh*Ww, H/Wh*W/Ww
        # print("[unfolding]: x.shape: ",x.shape)
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
        x_ours = windows
    else:

        # -- ours --
        # print("n_h,n_w: ",n_h,n_w)
        n_total = B * nH * nW
        x_ours = rearrange(x,'t h w c -> t c h w').contiguous()
        x_ours = unfolder(x_ours,0,n_total)
        shape_str = "n 1 1 c ph pw -> n ph pw c"
        x_ours = rearrange(x_ours,shape_str).contiguous()

    return x_ours

def impl_window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=stride) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        windows = windows.view(B, nH, nW, win_size, win_size, -1)
        windows = rearrange(windows,'n nh nw ph pw c -> (n nh nw) 1 1 c ph pw')
        windows = windows.contiguous()

        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    # print("windows.shape: ",windows.shape)
    return windows

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#                      Window Reverse
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def window_reverse(windows, win_size, H, W, dilation_rate=1, stride=None, region=None):
    if region is None:
        return impl_window_reverse_region(windows, win_size, H, W,
                                          dilation_rate, stride, region)
        # return impl_window_reverse(windows, win_size, H, W, dilation_rate)
    else:
        return impl_window_reverse_region(windows, win_size, H, W,
                                          dilation_rate, stride, region)

def impl_window_reverse_region(windows, win_size, H, W,
                               dilation_rate=1, stride=None, region=None):
    # B' ,Wh ,Ww ,C
    stride = win_size if stride is None else stride
    _,wh,ww,C = windows.shape
    T = 32
    # stride = win_size
    adj = stride//2
    only_full = True
    coords = None if region is None else region[2:]
    folder = dnls.ifold.iFold((T,C,H,W),coords,stride=stride,
                              dilation=1,adj=adj,only_full=only_full)
    wfolder = dnls.ifold.iFold((T,C,H,W),coords,stride=stride,
                               dilation=1,adj=adj,only_full=only_full)

    # -- compute shapes --
    dil = dilation_rate
    ps = win_size
    nH = (H - (ps-1)*dil - 1)//stride + 1
    nW = (W - (ps-1)*dil - 1)//stride + 1
    B = int(windows.shape[0] / (nH * nW))

    if dilation_rate !=1:

        x = windows.view(B,nH*nW,-1).permute(0,2,1)
        x = rearrange(x,'b (ph pw c) n -> b (c ph pw) n',ph=win_size,pw=win_size)
        x_ours = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate,
                        padding=4*(dilation_rate-1),stride=stride)
        x_ours = x_ours.permute(0,2,3,1)

    else:

        windows = windows.view(B, nH, nW, win_size, win_size, -1)
        windows = rearrange(windows,'n nh nw ph pw c -> (n nh nw) 1 1 c ph pw')
        windows = windows.contiguous()
        ones = th.ones_like(windows)
        x_ours = folder(windows,0)
        w_ours = wfolder(ones,0)
        # args = th.where(w_ours < 1)
        # print(args)
        # exit(0)
        x_ours = x_ours / (w_ours+1e-8)
        x_ours = rearrange(x_ours,'t c h w -> t h w c')

    return x_ours

def impl_window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1 or True:
        dil = dilation_rate
        ps = win_size
        nH = (H - (ps-1)*dil - 1)//stride + 1
        nW = (W - (ps-1)*dil - 1)//stride + 1

        x = windows.view(B,nH*nW,-1).permute(0,2,1)
        x = rearrange(x,'b (h w c) n -> b (c h w) n',h=win_size,w=win_size)
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate,
                   padding=4*(dilation_rate-1),stride=stride)
        x = x.permute(0,2,3,1)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
