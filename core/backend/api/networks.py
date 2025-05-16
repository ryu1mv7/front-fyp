import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image # Needed for Resize interpolation if used
import numpy as np # Needed for init_weights if using random

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
PATH_T1T2  = os.path.join(MODEL_DIR, 'set_200generator.pth') # T1→T2
PATH_T2T1  = os.path.join(MODEL_DIR, 'cgan_unet100generator.pth') # T2→T1 (NEWLY ADDED MODEL PATH)
PATH_PD2T2 = os.path.join(MODEL_DIR, 'best_gan_model.pth')  # PD→T2

# ─── Weight Initialization (Used by GeneratorT1T2 and GeneratorT2T1) ───────────
def init_weights(net, init_type='normal', gain=0.02):
    """
    Apply the recommended weight initialisation for GANs.
    """
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

# ─── Generator for T1->T2 ──────────────────────────────────────────────────────
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

        self.output = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        init_weights(self, init_type='normal', gain=0.02)

    def _conv_block(self, in_channels, out_channels,
                    kernel_size=4, stride=2, padding=1, batch_norm=True):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size, stride, padding,
                                bias=not batch_norm))
        if batch_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _deconv_block(self, in_channels, out_channels,
                      kernel_size=4, stride=2, padding=1, dropout=False):
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size, stride, padding,
                                         bias=False))
        layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(True))
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        d8 = self.d8(torch.cat([d7, e1], 1))
        return self.output(d8)

# ─── Generator for T2->T1 (from fyp-t1-t2.ipynb) ───────────────────────────────
class GeneratorT2T1(nn.Module):
    """
    A U-Net style generator for T2->T1 translation, with conditional input.
    Uses InstanceNorm2d. Output size matches input size.
    """
    def __init__(self, input_channels=1, output_channels=1):
        super(GeneratorT2T1, self).__init__()

        # Encoder layers
        self.e1 = self._conv_block(input_channels + 1, 64, batch_norm=False)  # +1 for condition channel
        self.e2 = self._conv_block(64, 128)
        self.e3 = self._conv_block(128, 256)
        self.e4 = self._conv_block(256, 512)
        self.e5 = self._conv_block(512, 512)
        self.e6 = self._conv_block(512, 512)
        self.e7 = self._conv_block(512, 512)
        self.e8 = self._conv_block(512, 512, batch_norm=False) # No batch norm on last encoder

        # Decoder layers
        self.d1 = self._deconv_block(512, 512, dropout=True)
        self.d2 = self._deconv_block(512 + 512, 512, dropout=True)
        self.d3 = self._deconv_block(512 + 512, 512, dropout=True)
        self.d4 = self._deconv_block(512 + 512, 512)
        self.d5 = self._deconv_block(512 + 512, 512)
        self.d6 = self._deconv_block(512 + 256, 256)
        self.d7 = self._deconv_block(256 + 128, 128)
        self.d8 = self._deconv_block(128 + 64, 64)

        # Output layer (preserves 256x256)
        self.output = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh() # Output normalized to [-1, 1]
        )
        init_weights(self, init_type='normal', gain=0.02)

    def _conv_block(self, in_channels, out_channels,
                    kernel_size=4, stride=2, padding=1, batch_norm=True):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size, stride, padding,
                                bias=not batch_norm))
        if batch_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _deconv_block(self, in_channels, out_channels,
                      kernel_size=4, stride=2, padding=1, dropout=False):
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size, stride, padding,
                                         bias=False))
        layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(True))
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x, label=0): # label defaults to 0 (e.g., non-tumor style)
        # Create a label tensor matching x's spatial dimensions and batch size
        # x shape: (B, C_in, H, W)
        # label_tensor shape: (B, 1, H, W)
        label_tensor = torch.ones_like(x[:, :1, :, :]) * label
        
        # Encoder pass with concatenated input and label
        e1 = self.e1(torch.cat([x, label_tensor], dim=1))
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        # Decoder pass with skip connections
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        d8 = self.d8(torch.cat([d7, e1], 1))
        
        return self.output(d8)

# ─── U-Net Blocks (For GeneratorPDT2) ──────────────────────────────────────────
class UNetDownPDT2(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize:    layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:      layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class UNetUpPDT2(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout: layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat((x, skip), dim=1)

# ─── Generator for PD->T2 ──────────────────────────────────────────────────────
class GeneratorPDT2(nn.Module):
    def __init__(self, input_channels=1, output_channels=1): # Standardized __init__
        super().__init__()
        self.down1      = UNetDownPDT2(input_channels,   64,  normalize=False) # Use input_channels
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
            nn.ConvTranspose2d(128, output_channels, 4, 2, 1), # Use output_channels
            nn.Tanh()
        )
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

# ─── Loader ────────────────────────────────────────────────────────────────────
def load_generator(path, model_class, strict=False):
    try:
        ckpt = torch.load(path, map_location='cpu')
        if isinstance(ckpt, dict) and 'generator_state_dict' in ckpt:
             state = ckpt['generator_state_dict']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
             state = ckpt['state_dict']
        elif isinstance(ckpt, dict):
             state = ckpt
        else:
             raise TypeError(f"Loaded object is not a dictionary: {type(ckpt)}")
    except Exception as e:
        print(f"Error loading or processing state dict from {path}: {e}")
        raise

    model = model_class(input_channels=1, output_channels=1) # Standardized instantiation

    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError as e:
         print(f"Error loading state_dict into {model_class.__name__} architecture: {e}")
         print("This often means the architecture defined in the code doesn't match the saved weights.")
         print(f"strict={strict} was used.")
         if not strict:
             print("Try strict=True to pinpoint missing/unexpected keys.")
         raise
    except Exception as e:
        print(f"An unexpected error occurred during model.load_state_dict: {e}")
        raise

    model.eval()
    return model

# Instantiate all generators
model_t1_t2 = load_generator(PATH_T1T2, GeneratorT1T2, strict=True)
model_t2_t1 = load_generator(PATH_T2T1, GeneratorT2T1, strict=True) # Added this line
model_pd2t2 = load_generator(PATH_PD2T2, GeneratorPDT2, strict=True)

# ─── Transforms ─────────────────────────────────────────────────────────────────
preprocess  = transforms.Compose([
    transforms.Resize((256,256), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
postprocess = transforms.Compose([
    transforms.Normalize([-1], [2]),
    transforms.ToPILImage()
])
