
# -- torch imports --
import torch
import torch.nn as nn

# -- timm imports --
from timm.models.layers import trunc_normal_

# -- local modules --
from .proj_modules import InputProj,OutputProj
from .lewin_transformer import BasicUformerLayer
from .scaling_modules import Downsample,Upsample

class Uformer(nn.Module):
    def __init__(self, img_size=128, in_chans=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='ffn',
                 se_layer=False, dowsample=Downsample, upsample=Upsample,
                 fwd_mode="dnls_k", stride=None, ws=-1, wt=0, k=100, sb=None,
                 **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size =win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.verbose = False

        # -- modified params --
        self.fwd_mode = fwd_mode
        self.stride = stride
        self.ws = ws
        self.wt = wt
        self.k = k
        self.sb = sb

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate]*depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim,
                                    kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2*embed_dim,
                                      out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                            output_dim=embed_dim,
                            input_resolution=(img_size,img_size),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,
                            token_mlp=token_mlp,
                            se_layer=se_layer,
                            fwd_mode=fwd_mode,stride=stride,ws=ws,wt=wt,k=k,sb=sb)
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size // 2,img_size // 2),
                            depth=depths[1],
                            num_heads=num_heads[1],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,
                            token_mlp=token_mlp,se_layer=se_layer,
                            fwd_mode=fwd_mode,stride=stride,ws=ws,wt=wt,k=k,sb=sb)
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size//(2 ** 2),img_size//(2 ** 2)),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,
                            token_mlp=token_mlp,se_layer=se_layer,
                            fwd_mode=fwd_mode,stride=stride,ws=ws,wt=wt,k=k,sb=sb)
        self.dowsample_2 = dowsample(embed_dim*4, embed_dim*8)
        self.encoderlayer_3 = BasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            input_resolution=(img_size//(2 ** 3),img_size//(2 ** 3)),
                            depth=depths[3],
                            num_heads=num_heads[3],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,
                            token_mlp=token_mlp,se_layer=se_layer,
                            fwd_mode=fwd_mode,stride=stride,ws=ws,wt=wt,k=k,sb=sb)
        self.dowsample_3 = dowsample(embed_dim*8, embed_dim*16)

        # Bottleneck
        self.conv = BasicUformerLayer(dim=embed_dim*16,
                            output_dim=embed_dim*16,
                            input_resolution=(img_size//(2 ** 4),img_size//(2 ** 4)),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,
                            token_mlp=token_mlp,se_layer=se_layer,
                            fwd_mode=fwd_mode,stride=stride,ws=ws,wt=wt,k=k,sb=sb)

        # Decoder
        self.upsample_0 = upsample(embed_dim*16, embed_dim*8)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim*16,
                            output_dim=embed_dim*16,
                            input_resolution=(img_size//(2 ** 3),img_size//(2 ** 3)),
                            depth=depths[5],
                            num_heads=num_heads[5],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[:depths[5]],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,
                            token_mlp=token_mlp,se_layer=se_layer,
                            fwd_mode=fwd_mode,stride=stride,ws=ws,wt=wt,k=k,sb=sb)
        self.upsample_1 = upsample(embed_dim*16, embed_dim*4)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            input_resolution=(img_size//(2 ** 2),img_size//(2 ** 2)),
                            depth=depths[6],
                            num_heads=num_heads[6],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,
                            token_mlp=token_mlp,se_layer=se_layer,
                            fwd_mode=fwd_mode,stride=stride,ws=ws,wt=wt,k=k,sb=sb)
        self.upsample_2 = upsample(embed_dim*8, embed_dim*2)
        self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size//2,img_size//2),
                            depth=depths[7],
                            num_heads=num_heads[7],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,
                            token_mlp=token_mlp,se_layer=se_layer,
                            fwd_mode=fwd_mode,stride=stride,ws=ws,wt=wt,k=k,sb=sb)
        self.upsample_3 = upsample(embed_dim*4, embed_dim)
        self.decoderlayer_3 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size,img_size),
                            depth=depths[8],
                            num_heads=num_heads[8],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,
                            token_mlp=token_mlp,se_layer=se_layer,
                            fwd_mode=fwd_mode,stride=stride,ws=ws,wt=wt,k=k,sb=sb)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def vprint(self,*args,**kwargs):
        if self.verbose:
            print(*args,**kwargs)

    def forward(self, x, flows=None, mask=None, region=None):
        # Input Projection

        t,c,H,W = x.shape
        y = self.input_proj(x)
        y = self.pos_drop(y)

        # -- Encoder --
        h,w = H,W
        conv0 = self.encoderlayer_0(y,h,w,flows=flows,mask=mask,region=region)
        if not(self.training): del y
        self.vprint("conv0.shape: ",conv0.shape)
        pool0 = self.dowsample_0(conv0,h,w)
        self.vprint("pool0.shape: ",pool0.shape)

        h,w = h//2,w//2
        conv1 = self.encoderlayer_1(pool0,h,w,flows=flows,mask=mask,region=region)
        if not(self.training): del pool0
        self.vprint("conv1.shape: ",conv1.shape)
        pool1 = self.dowsample_1(conv1,h,w)

        h,w = h//2,w//2
        conv2 = self.encoderlayer_2(pool1,h,w,flows=flows,mask=mask,region=region)
        if not(self.training): del pool1
        self.vprint("conv2.shape: ",conv2.shape)
        pool2 = self.dowsample_2(conv2,h,w)

        h,w = h//2,w//2
        conv3 = self.encoderlayer_3(pool2,h,w,flows=flows,mask=mask,region=region)
        if not(self.training): del pool2
        self.vprint("conv3.shape: ",conv3.shape)
        pool3 = self.dowsample_3(conv3,h,w)

        # Bottleneck
        h,w = h//2,w//2
        conv4 = self.conv(pool3, h, w, flows=flows, mask=mask,region=region)
        if not(self.training): del pool3
        self.vprint("conv4.shape: ",conv4.shape)

        #Decoder
        up0 = self.upsample_0(conv4,h,w)
        h,w = 2*h,2*w
        deconv0 = torch.cat([up0,conv3],-1)
        deconv0 = self.decoderlayer_0(deconv0,h,w,flows=flows,mask=mask,region=region)
        self.vprint("deconv0.shape: ",deconv0.shape)
        torch.cuda.empty_cache()

        up1 = self.upsample_1(deconv0,h,w)
        if not(self.training): del deconv0,conv3,conv4,up0
        torch.cuda.empty_cache()
        h,w = 2*h,2*w
        deconv1 = torch.cat([up1,conv2],-1)
        deconv1 = self.decoderlayer_1(deconv1,h,w,flows=flows,mask=mask,region=region)
        self.vprint("deconv1.shape: ",deconv1.shape)
        torch.cuda.empty_cache()

        up2 = self.upsample_2(deconv1,h,w)
        if not(self.training): del deconv1,conv2,up1
        torch.cuda.empty_cache()
        h,w = 2*h,2*w
        deconv2 = torch.cat([up2,conv1],-1)
        deconv2 = self.decoderlayer_2(deconv2,h,w,flows=flows,mask=mask,region=region)
        self.vprint("deconv2.shape: ",deconv2.shape)
        torch.cuda.empty_cache()

        up3 = self.upsample_3(deconv2,h,w)
        if not(self.training): del deconv2,conv1,up2
        torch.cuda.empty_cache()
        h,w = 2*h,2*w
        deconv3 = torch.cat([up3,conv0],-1)
        deconv3 = self.decoderlayer_3(deconv3,h,w,flows=flows,mask=mask,region=region)
        self.vprint("deconv3.shape: ",deconv3.shape)
        torch.cuda.empty_cache()

        # Output Projection
        y = self.output_proj(deconv3,h,w)
        return x + y

    def flops(self):
        flops = 0
        # Input Projection
        flops += self.input_proj.flops(self.reso,self.reso)
        # Encoder
        flops += self.encoderlayer_0.flops()+self.dowsample_0.flops(self.reso,self.reso)
        flops += self.encoderlayer_1.flops()+self.dowsample_1.flops(self.reso//2,self.reso//2)
        flops += self.encoderlayer_2.flops()+self.dowsample_2.flops(self.reso//2**2,self.reso//2**2)
        flops += self.encoderlayer_3.flops()+self.dowsample_3.flops(self.reso//2**3,self.reso//2**3)

        # Bottleneck
        flops += self.conv.flops()

        # Decoder
        flops += self.upsample_0.flops(self.reso//2**4,self.reso//2**4)+self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(self.reso//2**3,self.reso//2**3)+self.decoderlayer_1.flops()
        flops += self.upsample_2.flops(self.reso//2**2,self.reso//2**2)+self.decoderlayer_2.flops()
        flops += self.upsample_3.flops(self.reso//2,self.reso//2)+self.decoderlayer_3.flops()

        # Output Projection
        flops += self.output_proj.flops(self.reso,self.reso)
        return flops
