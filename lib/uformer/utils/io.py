# -- linalg --
import torch as th
import numpy as np
from einops import rearrange

# -- file io --
from PIL import Image
from pathlib import Path

def save_burst(burst,root,name,fstart=0):

    # -- path --
    root = Path(str(root))
    if not root.exists():
        print(f"Making dir for save_burst [{str(root)}]")
        root.mkdir(parents=True)
    assert root.exists()

    # -- save --
    save_fns = []
    nframes = burst.shape[0]
    for t in range(nframes):
        fid = t + fstart
        img_t = burst[t]
        path_t = root / ("%s_%05d.png" % (name,fid))
        save_image(img_t,str(path_t))
        save_fns.append(str(path_t))
    return save_fns

def save_image(image,path):

    # -- to numpy --
    if th.is_tensor(image):
        image = image.detach().cpu().numpy()

    # -- rescale --
    if image.max() > 500: # probably from a fold
        image /= image.max()

    # -- to uint8 --
    if image.max() < 100:
        image = image*255.
    image = np.clip(image,0,255).astype(np.uint8)

    # -- save --
    image = rearrange(image,'c h w -> h w c')
    img = Image.fromarray(image)
    img.save(path)
