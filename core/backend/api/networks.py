import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# ==============================================
# Path Configuration
# ==============================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

MODEL_PATHS = {
    't1_to_t2': os.path.join(MODEL_DIR, 'set_200generator.pth'),
    't2_to_t1': os.path.join(MODEL_DIR, 'cgan_unet100generator.pth'),
    'pd_to_t2': os.path.join(MODEL_DIR, 'best_gan_model.pth')
}

# ==============================================
# Utility Functions
# ==============================================
def init_weights(net, init_type='normal', gain=0.02):
    """Weight initialization for GANs"""
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
                raise NotImplementedError(f"Initialization method [{init_type}] not implemented")
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
                
        elif 'BatchNorm2d' in classname or 'InstanceNorm2d' in classname:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    
    net.apply(init_func)

# ==============================================
# Model Components
# ==============================================
class ConvBlock(nn.Module):
    """Standard convolutional block with optional batch norm"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, 
                 batch_norm=True, activation='leaky_relu'):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                     bias=not batch_norm)
        ]
        
        if batch_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
            
        if activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
            
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class DeconvBlock(nn.Module):
    """Standard deconvolutional block with optional dropout"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, 
                 dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, 
                              bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        if dropout:
            layers.append(nn.Dropout(0.5))
            
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

# ==============================================
# Generator Models
# ==============================================
class T1ToT2Generator(nn.Module):
    """U-Net generator for T1 to T2 translation"""
    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()
        
        # Encoder
        self.e1 = ConvBlock(input_channels, 64, batch_norm=False)
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        self.e4 = ConvBlock(256, 512)
        self.e5 = ConvBlock(512, 512)
        self.e6 = ConvBlock(512, 512)
        self.e7 = ConvBlock(512, 512)
        self.e8 = ConvBlock(512, 512, batch_norm=False)
        
        # Decoder
        self.d1 = DeconvBlock(512, 512, dropout=True)
        self.d2 = DeconvBlock(1024, 512, dropout=True)
        self.d3 = DeconvBlock(1024, 512, dropout=True)
        self.d4 = DeconvBlock(1024, 512)
        self.d5 = DeconvBlock(1024, 512)
        self.d6 = DeconvBlock(768, 256)
        self.d7 = DeconvBlock(384, 128)
        self.d8 = DeconvBlock(192, 64)
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
        init_weights(self)

    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        
        # Decoder with skip connections
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        d8 = self.d8(torch.cat([d7, e1], 1))
        
        return self.output(d8)

class T2ToT1Generator(nn.Module):
    """Conditional U-Net generator for T2 to T1 translation"""
    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()
        
        # Encoder
        self.e1 = ConvBlock(input_channels + 1, 64, batch_norm=False)  # +1 for condition channel
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        self.e4 = ConvBlock(256, 512)
        self.e5 = ConvBlock(512, 512)
        self.e6 = ConvBlock(512, 512)
        self.e7 = ConvBlock(512, 512)
        self.e8 = ConvBlock(512, 512, batch_norm=False)
        
        # Decoder
        self.d1 = DeconvBlock(512, 512, dropout=True)
        self.d2 = DeconvBlock(1024, 512, dropout=True)
        self.d3 = DeconvBlock(1024, 512, dropout=True)
        self.d4 = DeconvBlock(1024, 512)
        self.d5 = DeconvBlock(1024, 512)
        self.d6 = DeconvBlock(768, 256)
        self.d7 = DeconvBlock(384, 128)
        self.d8 = DeconvBlock(192, 64)
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
        init_weights(self)

    def forward(self, x, label=0):
        # Create condition tensor
        label_tensor = torch.ones_like(x[:, :1, :, :]) * label
        
        # Encoder
        e1 = self.e1(torch.cat([x, label_tensor], dim=1))
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        
        # Decoder with skip connections
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        d8 = self.d8(torch.cat([d7, e1], 1))
        
        return self.output(d8)

class PDToT2Generator(nn.Module):
    """U-Net generator for PD to T2 translation"""
    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()
        
        # Downsampling
        self.down1 = ConvBlock(input_channels, 64, batch_norm=True, activation='leaky_relu')
        self.down2 = ConvBlock(64, 128, batch_norm=True, activation='leaky_relu')
        self.down3 = ConvBlock(128, 256, batch_norm=True, activation='leaky_relu')
        self.down4 = ConvBlock(256, 512, batch_norm=True, activation='leaky_relu', dropout=0.5)
        self.down5 = ConvBlock(512, 512, batch_norm=True, activation='leaky_relu', dropout=0.5)
        self.down6 = ConvBlock(512, 512, batch_norm=True, activation='leaky_relu', dropout=0.5)
        self.down7 = ConvBlock(512, 512, batch_norm=True, activation='leaky_relu', dropout=0.5)
        self.bottleneck = ConvBlock(512, 512, batch_norm=False, activation='relu')
        
        # Upsampling
        self.up1 = DeconvBlock(512, 512, dropout=True)
        self.up2 = DeconvBlock(1024, 512, dropout=True)
        self.up3 = DeconvBlock(1024, 512, dropout=True)
        self.up4 = DeconvBlock(1024, 512)
        self.up5 = DeconvBlock(1024, 256)
        self.up6 = DeconvBlock(512, 128)
        self.up7 = DeconvBlock(256, 64)
        
        # Output
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(128, output_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bn = self.bottleneck(d7)
        
        # Decoder with skip connections
        u1 = self.up1(bn)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        
        return self.final_up(torch.cat([u7, d1], 1))

# ==============================================
# Model Loading and Initialization
# ==============================================
def load_generator(path, model_class, strict=False):
    """Load generator model from checkpoint"""
    try:
        ckpt = torch.load(path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(ckpt, dict):
            if 'generator_state_dict' in ckpt:
                state = ckpt['generator_state_dict']
            elif 'state_dict' in ckpt:
                state = ckpt['state_dict']
            else:
                state = ckpt
        else:
            raise TypeError(f"Loaded object is not a dictionary: {type(ckpt)}")
            
    except Exception as e:
        raise RuntimeError(f"Error loading state dict from {path}: {e}")

    # Initialize model
    model = model_class(input_channels=1, output_channels=1)
    
    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError as e:
        raise RuntimeError(f"Error loading state_dict into {model_class.__name__}: {e}")
    
    model.eval()
    return model

# Initialize all models
MODELS = {}

try:
    MODELS['t1_to_t2'] = load_generator(MODEL_PATHS['t1_to_t2'], T1ToT2Generator, strict=True)
except Exception as e:
    print("Skipping t1_to_t2:", e)

try:
    MODELS['t2_to_t1'] = load_generator(MODEL_PATHS['t2_to_t1'], T2ToT1Generator, strict=True)
except Exception as e:
    print("Skipping t2_to_t1:", e)

try:
    MODELS['pd_to_t2'] = load_generator(MODEL_PATHS['pd_to_t2'], PDToT2Generator, strict=True)
except Exception as e:
    print("Skipping pd_to_t2:", e)


# ==============================================
# Image Processing
# ==============================================
preprocess = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

postprocess = transforms.Compose([
    transforms.Normalize([-1], [2]),
    transforms.ToPILImage()
])

# ==============================================
# Additional Model (Checkpoint Matching)
# ==============================================
class UNet2DGeneratorCheckpointMatch(nn.Module):
    """Alternative U-Net architecture for checkpoint compatibility"""
    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU()
        )
        
        # Upsampling
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1)
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        bn = self.bottleneck(d3)
        
        u3 = self.up3(bn)
        u2 = self.up2(torch.cat([u3, d3], 1))
        u1 = self.up1(torch.cat([u2, d2], 1))
        
        return self.final(torch.cat([u1, d1], 1))