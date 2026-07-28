"""
nafnet_mri.py

NAFNet-based MRI enhancement module adapted from:
Chen et al., "Simple Baselines for Image Restoration", ECCV 2022

Features:
- Preserves original NAFNet design (SimpleGate, SCA, beta/gamma)
- Optional flat (non–U-Net) architecture for fixed-resolution MRI
- Conservative global residual scaling for medical imaging

Intended use:
- MRI enhancement prior to downstream tasks (e.g. classification)

Note:
This file defines models only. Training logic lives elsewhere.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# LAYER NORMALIZATION
class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean((2, 3), keepdim=True)
        var = x.var((2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

# SIMPLE GATE
class SimpleGate(nn.Module):
    """NAF activation: split channels and multiply"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

# NAF BLOCK
class NAFBlock(nn.Module):
    """
    Original NAFBlock from Chen et al. 2022
    
    COMPONENTS (ALL PRESERVED):
    1. Spatial convolutions with depthwise separable
    2. SimpleGate activation
    3. Simplified Channel Attention (SCA) ← IMPORTANT!
    4. Feed-Forward Network (FFN)
    5. Learnable residual scaling (beta, gamma)
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        
        # ---- Spatial Processing ----
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channel, dw_channel, 3, 
            padding=1, groups=dw_channel, bias=True
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, bias=True)
        
        # ---- Simplified Channel Attention (CRITICAL - DARI ORIGINAL!) ----
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True),
        )
        
        # ---- Feed-Forward Network ----
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, bias=True)
        
        # ---- Normalization ----
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        
        # ---- SimpleGate ----
        self.sg = SimpleGate()
        
        # ---- Dropout ----
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        
        # ---- Learnable Residual Scaling (DARI ORIGINAL) ----
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    
    def forward(self, inp):
        # ---- Spatial Branch ----
        x = self.norm1(inp)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        
        x = self.dropout1(x)
        
        y = inp + x * self.beta
        
        # ---- FFN Branch ----
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        
        x = self.dropout2(x)
        
        return y + x * self.gamma

# NAFNET - FULL ARCHITECTURE
class NAFNet(nn.Module):
    """
    Full NAFNet with U-Net structure (ORIGINAL dari paper)
    
    Multi-scale encoder-decoder dengan skip connections
    """
    def __init__(
        self,
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[]
    ):
        super().__init__()
        
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = width
        
        # Encoder
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2
        
        # Middle
        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )
        
        # Decoder
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
        
        self.padder_size = 2 ** len(self.encoders)
    
    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        
        x = self.intro(inp)
        
        encs = []
        
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        
        x = self.middle_blks(x)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        
        x = self.ending(x)
        x = x + inp
        
        return x[:, :, :H, :W]
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

# MEDICAL WRAPPER (modifikasi untuk medical imaging)
class MedicalNAFNet(nn.Module):
    """
    MODIFIKASI: Wrapper dengan conservative global residual scaling
    
    Original NAFNet output: inp + residual
    Medical version: inp + α * residual (dengan α=0.1-0.2)
    
    Rationale:
    - Medical images require conservative enhancement
    - Preserve diagnostic features > visual quality
    - Avoid over-correction yang bisa distort pathology
    """
    def __init__(self, base_model, residual_scale=0.15):
        super().__init__()
        self.base_model = base_model
        self.residual_scale = residual_scale
    
    def forward(self, inp):
        # Base model sudah return inp + residual
        out = self.base_model(inp)
        
        # Extract residual
        residual = out - inp
        
        # Apply conservative scaling
        return inp + self.residual_scale * residual

# FACTORY FUNCTIONS (CONVENIENCE untuk medical imaging)

def create_flat_nafnet(
    img_channel=1,
    width=32,
    num_blocks=6,
    drop_out_rate=0.
):
    """
    MODIFIKASI: Simplified flat architecture (NO U-Net)
    
    Menggunakan NAFBlock original (dengan SCA) tapi dalam flat configuration
    Cocok untuk medical imaging dengan fixed resolution
    """
    # Build flat model manually
    class FlatNAFNet(nn.Module):
        def __init__(self, img_channel, width, num_blocks, drop_out_rate):
            super().__init__()
            self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
            self.blocks = nn.Sequential(*[
                NAFBlock(width, drop_out_rate=drop_out_rate) 
                for _ in range(num_blocks)
            ])
            self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)
        
        def forward(self, inp):
            x = self.intro(inp)
            x = self.blocks(x)
            x = self.ending(x)
            return inp + x
    
    return FlatNAFNet(img_channel, width, num_blocks, drop_out_rate)

def create_medical_nafnet(
    img_channel=1,
    width=32,
    num_blocks=6,
    residual_scale=0.15,
    drop_out_rate=0.,
    use_unet=False,
    middle_blk_num=4,
    enc_blk_nums=[1, 1, 2, 4],
    dec_blk_nums=[1, 1, 1, 1]
):
    """
    Main factory function untuk medical MRI enhancement
    
    Parameters:
    -----------
    use_unet : bool
        True = Full NAFNet (multi-scale U-Net)
        False = Flat NAFNet (simplified)
    residual_scale : float
        Conservative scaling (0.1-0.2 recommended)
    """
    if use_unet:
        # Full NAFNet dengan U-Net
        base_model = NAFNet(
            img_channel=img_channel,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blk_nums,
            dec_blk_nums=dec_blk_nums
        )
    else:
        # Flat NAFNet (simplified)
        base_model = create_flat_nafnet(
            img_channel=img_channel,
            width=width,
            num_blocks=num_blocks,
            drop_out_rate=drop_out_rate
        )
    
    # Wrap dengan conservative scaling
    return MedicalNAFNet(base_model, residual_scale=residual_scale)

def NAFNet_flat(img_channel=1, width=32, num_blocks=4, residual_scale=0.03):
    """
    DEPRECATED: Gunakan create_medical_nafnet() instead
    Kept untuk backward compatibility
    """
    return create_medical_nafnet(
        img_channel=img_channel,
        width=width,
        num_blocks=num_blocks,
        residual_scale=residual_scale,
        use_unet=False
    )

# USAGE EXAMPLES
if __name__ == "__main__":
    print("=" * 70)
    print("NAFNet Medical MRI Enhancement - Architecture Options")
    print("=" * 70)
    
    # Option 1: Flat NAFNet (Recommended untuk start)
    print("\n1. Flat NAFNet (Simplified, with SCA)")
    model_flat = create_medical_nafnet(
        img_channel=1,
        width=32,
        num_blocks=6,
        residual_scale=0.15,
        use_unet=False
    )
    
    x = torch.randn(1, 1, 224, 224)
    y = model_flat(x)
    
    params_flat = sum(p.numel() for p in model_flat.parameters())
    print(f"   Parameters: {params_flat:,}")
    print(f"   Input:  {x.shape}")
    print(f"   Output: {y.shape}")
    
    # Verify SCA present
    has_sca = any(hasattr(m, 'sca') for m in model_flat.modules() if isinstance(m, NAFBlock))
    print(f"   SCA Present: {has_sca} ✓" if has_sca else f"   SCA Present: {has_sca} ✗")
    
    # Option 2: Full NAFNet (U-Net)
    print("\n2. Full NAFNet (Multi-scale U-Net, with SCA)")
    model_full = create_medical_nafnet(
        img_channel=1,
        width=32,
        middle_blk_num=4,
        enc_blk_nums=[1, 1, 2, 4],
        dec_blk_nums=[1, 1, 1, 1],
        residual_scale=0.15,
        use_unet=True
    )
    
    y = model_full(x)
    
    params_full = sum(p.numel() for p in model_full.parameters())
    print(f"   Parameters: {params_full:,}")
    print(f"   Input:  {x.shape}")
    print(f"   Output: {y.shape}")
    
    # Verify SCA present
    has_sca = any(hasattr(m, 'sca') for m in model_full.modules() if isinstance(m, NAFBlock))
    print(f"   SCA Present: {has_sca} ✓" if has_sca else f"   SCA Present: {has_sca} ✗")
    
    print("\n" + "=" * 70)
    print("KEY FEATURES (Preserved from NAFNet original):")
    print("  ✓ SimpleGate activation (NAF principle)")
    print("  ✓ Simplified Channel Attention (SCA)")
    print("  ✓ Learnable residual scaling (beta, gamma)")
    print("  ✓ LayerNorm2d normalization")
    print("\nMEDICAL ADAPTATIONS:")
    print("  + Conservative global residual scaling (α=0.15)")
    print("  + Optional flat architecture (no U-Net)")
    print("=" * 70)