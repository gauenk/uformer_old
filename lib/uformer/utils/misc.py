
import torch as th
import pickle

def optional(pydict,key,default):
    if pydict is None: return default
    elif key in pydict: return pydict[key]
    else: return default

def optional_delete(pydict,key):
    if pydict is None: return
    elif key in pydict: del pydict[key]
    else: return

def tuple_as_int(elem):
    if hasattr(elem,"__getitem__"):
        return elem[0]
    else:
        return elem

def assert_nonan(tensor):
    assert th.any(th.isnan(tensor)).item() is False

def rslice_pair(vid_a,vid_b,coords):
    vid_a = rslice(vid_a,coords)
    vid_b = rslice(vid_b,coords)
    return vid_a,vid_b

def rslice(vid,coords):
    if coords is None: return vid
    if len(coords) == 0: return vid
    if th.is_tensor(coords):
        coords = coords.type(th.int)
        coords = list(coords.cpu().numpy())
    fs,fe,t,l,b,r = coords
    return vid[fs:fe,:,t:b,l:r]

def write_pickle(fn,obj):
    with open(str(fn),"wb") as f:
        pickle.dump(obj,f)

def read_pickle(fn):
    with open(str(fn),"rb") as f:
        obj = pickle.load(f)
    return obj
