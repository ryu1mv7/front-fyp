import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PATH_T1T2 = os.path.join(MODEL_DIR, 'set_200generator.pth')  # T1→T2
PATH_PD2T2 = os.path.join(MODEL_DIR, 'best_gan_model.pth')  # PD→T2
PATH_BRATS_T2F_SEG = os.path.join(MODEL_DIR, 'cgan_models_t2f_seg_250.pth')  # Your addition
PATH_MRI2CT = os.path.join(MODEL_DIR, 'pix2pix_weights.pth')  # Your addition
PATH_IXI_BRAIN_SEG = os.path.join(MODEL_DIR, 'ixi_multiclass_model.pt')  # Your addition

# ─── Weight Initialization (From General Version) ──────────────────────────────
def init_weights(net, init_type='normal', gain=0.02):
    """Apply the recommended weight initialisation for GANs."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in classname or 'Linear' in classname):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f"initialisation method [{init_type}] is not implemented")
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname or 'InstanceNorm2d' in classname:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

# ─── Generator for T1->T2 (From General Version) ────────────────────────────────
class GeneratorT1T2(nn.Module):
    """
    A U-Net style generator with 8 down-samplings and 8 up-samplings. (From Notebook)
    Uses InstanceNorm2d and specific final layer.
    """
    def __init__(self, input_channels=1, output_channels=1):
        super(GeneratorT1T2, self).__init__() 

        # Encoder layers
        self.e1 = self._conv_block(input_channels, 64, batch_norm=False)  # 256 -> 128
        self.e2 = self._conv_block(64, 128)                               # 128 -> 64
        self.e3 = self._conv_block(128, 256)                              # 64 -> 32
        self.e4 = self._conv_block(256, 512)                              # 32 -> 16
        self.e5 = self._conv_block(512, 512)                              # 16 -> 8
        self.e6 = self._conv_block(512, 512)                              # 8 -> 4
        self.e7 = self._conv_block(512, 512)                              # 4 -> 2
        # No batch norm on last encoder
        self.e8 = self._conv_block(512, 512, batch_norm=False)            # 2 -> 1

        # Decoder layers
        self.d1 = self._deconv_block(512, 512, dropout=True)              # 1 -> 2
        self.d2 = self._deconv_block(512 + 512, 512, dropout=True)        # 2 -> 4
        self.d3 = self._deconv_block(512 + 512, 512, dropout=True)        # 4 -> 8
        self.d4 = self._deconv_block(512 + 512, 512)                      # 8 -> 16
        self.d5 = self._deconv_block(512 + 512, 512)                      # 16 -> 32
        self.d6 = self._deconv_block(512 + 256, 256)                      # 32 -> 64
        self.d7 = self._deconv_block(256 + 128, 128)                      # 64 -> 128
        self.d8 = self._deconv_block(128 + 64, 64)                        # 128 -> 256

        # Output
        self.output = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh() # Output normalized to [-1, 1]
        )

        # Weight initialisation (Will be overwritten by loaded state_dict, but good practice)
        init_weights(self, init_type='normal', gain=0.02) # Make sure init_weights exists

    def _conv_block(self, in_channels, out_channels,
                    kernel_size=4, stride=2, padding=1, batch_norm=True):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size, stride, padding,
                                bias=not batch_norm))
        if batch_norm:
            layers.append(nn.InstanceNorm2d(out_channels)) # Using InstanceNorm2d
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _deconv_block(self, in_channels, out_channels,
                      kernel_size=4, stride=2, padding=1, dropout=False):
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size, stride, padding,
                                         bias=False)) # Bias=False when using Norm layer after
        layers.append(nn.InstanceNorm2d(out_channels)) # Using InstanceNorm2d
        layers.append(nn.ReLU(True))
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder pass
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        # Decoder pass with skip connections
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], 1)) # Concatenate along channel dimension
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        d8 = self.d8(torch.cat([d7, e1], 1))

        # Output layer
        return self.output(d8)

# ─── U-Net Blocks (For GeneratorPDT2) ──────────────────────────────────────────
class UNetDownPDT2(nn.Module): # RENAMED
    def __init__(self, in_ch, out_ch, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize:    layers.append(nn.BatchNorm2d(out_ch)) # Using BatchNorm2d
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:      layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class UNetUpPDT2(nn.Module): # RENAMED
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch), # Using BatchNorm2d
            nn.ReLU(inplace=True),
        ]
        if dropout: layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat((x, skip), dim=1)

# ─── Generator for PD->T2 (Original GeneratorUNet) ─────────────────────────────
class GeneratorPDT2(nn.Module): # RENAMED from GeneratorUNet
    # CHANGE THE __init__ SIGNATURE HERE:
    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()
        # UPDATE ARGUMENT NAME USAGE HERE:
        self.down1      = UNetDownPDT2(input_channels,   64,  normalize=False)
        self.down2      = UNetDownPDT2(64,      128)
        self.down3      = UNetDownPDT2(128,     256)
        self.down4      = UNetDownPDT2(256,     512, dropout=0.5)
        self.down5      = UNetDownPDT2(512,     512, dropout=0.5)
        self.down6      = UNetDownPDT2(512,     512, dropout=0.5)
        self.down7      = UNetDownPDT2(512,     512, dropout=0.5)
        self.bottleneck = UNetDownPDT2(512,     512, normalize=False)
        self.up1        = UNetUpPDT2(512,   512, dropout=0.5)
        self.up2        = UNetUpPDT2(1024,  512, dropout=0.5)
        self.up3        = UNetUpPDT2(1024,  512, dropout=0.5)
        self.up4        = UNetUpPDT2(1024,  512, dropout=0.5)
        self.up5        = UNetUpPDT2(1024,  256)
        self.up6        = UNetUpPDT2(512,   128)
        self.up7        = UNetUpPDT2(256,   64)
        self.final_up   = nn.Sequential(
            # UPDATE ARGUMENT NAME USAGE HERE:
            nn.ConvTranspose2d(128, output_channels, 4, 2, 1),
            nn.Tanh()
        )
    # forward method remains the same...
    def forward(self, x):
        d1 = self.down1(x);  d2 = self.down2(d1)
        d3 = self.down3(d2); d4 = self.down4(d3)
        d5 = self.down5(d4); d6 = self.down6(d5)
        d7 = self.down7(d6); bn = self.bottleneck(d7)
        u1 = self.up1(bn, d7);   u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5);   u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3);   u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final_up(u7)

# ─── Your Custom Models ────────────────────────────────────────────────────────
class Pix2PixGenerator(nn.Module):
    """Your lightweight Pix2Pix model for MRI→CT."""
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features * 2, 4, 2, 1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(features, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

class CGANWithSegOutput(nn.Module):
    """Your model for BraTS (T2-Flair + Segmentation)."""
    def __init__(self, input_channels=3, output_channels=1):
        super().__init__()
        self.base = GeneratorT1T2(input_channels=input_channels, output_channels=output_channels)
        self.seg_head = nn.Sequential(
            nn.Conv2d(output_channels, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        t2f = self.base(x)
        seg = self.seg_head(t2f)
        return t2f, seg

class BrainSegNet(nn.Module):
    """Your brain segmentation model (IXI dataset)."""
    def __init__(self, input_channels=1, output_channels=4):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),   # [32, 1, 3, 3]
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),               # [32, 32, 3, 3]
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.final = nn.Conv2d(32, output_channels, kernel_size=1)

        self._load_weights()

    def _load_weights(self):
        weights = torch.load(PATH_IXI_BRAIN_SEG, map_location='cpu')
        self.load_state_dict(weights)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        b = self.bottleneck(e2)
        u2 = self.up2(b)
        d2 = self.decoder2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.decoder1(torch.cat([u1, e1], dim=1))
        return self.final(d1)

# ─── Loader (Improved from General Version) ─────────────────────────────────────
def load_generator(path, model_class, strict=False):
    """Load a generator with better error handling."""
    try:
        ckpt = torch.load(path, map_location='cpu')
        if isinstance(ckpt, dict) and 'generator_state_dict' in ckpt:
             state = ckpt['generator_state_dict']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt: # Common alternative key
             state = ckpt['state_dict']
        elif isinstance(ckpt, dict): # Assume it might be the state dict itself
             state = ckpt
        else: # Should not happen if .pth contains a dict or state_dict
             raise TypeError(f"Loaded object is not a dictionary: {type(ckpt)}")
    except Exception as e:
        print(f"Error loading or processing state dict from {path}: {e}")
        raise
    
    # Initialize model (handles both your and general version's args)
    if model_class.__name__ == "Pix2PixGenerator":
        model = model_class(in_channels=1, out_channels=1)
    else:
        model = model_class(input_channels=1, output_channels=1)  # General models

    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError as e:
        missing = [k for k in state.keys() if k not in model.state_dict()]
        unexpected = [k for k in model.state_dict() if k not in state.keys()]
        raise RuntimeError(
            f"Failed to load {model_class.__name__}:\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}"
        ) from e

    model.eval()
    return model

# ─── Transforms (From General Version) ──────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

postprocess = transforms.Compose([
    transforms.Normalize([-1], [2]),  # Reverse Normalize([0.5], [0.5])
    transforms.ToPILImage(),
])

# ─── Instantiate Models (Combined) ──────────────────────────────────────────────
model_t1_t2 = load_generator(PATH_T1T2, GeneratorT1T2, strict=True)
model_pd2t2 = load_generator(PATH_PD2T2, GeneratorPDT2, strict=True)
model_brats_t2f_seg = load_generator(PATH_BRATS_T2F_SEG, CGANWithSegOutput, strict=False)
model_mri2ct = load_generator(PATH_MRI2CT, Pix2PixGenerator, strict=False)