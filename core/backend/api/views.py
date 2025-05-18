import io
import base64
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from PIL import Image
import lpips
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import tempfile
import os
import nibabel as nib

from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .networks import MODELS, UNet2DGeneratorCheckpointMatch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Initialise LPIPS metric ---
lpips_fn = lpips.LPIPS(net='alex')

# --- Transforms ---
standard_preprocess = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # To [-1, 1]
])

preprocess_for_t2t1 = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.BICUBIC),
    transforms.ToTensor()  # To [0, 1]
])

standard_postprocess = transforms.Compose([
    transforms.Normalize([-1], [2]),  # From [-1, 1] to [0, 1]
    transforms.ToPILImage()
])

postprocess_for_t2t1 = transforms.Compose([
    transforms.ToPILImage()  # Input is already effectively in [0,1]
])

lpips_preprocess_for_lpips = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class ConvertImageView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        img_f = request.FILES.get('image')
        conv = request.data.get('conversionType')
        image_format = request.data.get('imageFormat', 'png')

        if not img_f or not conv:
            return Response({'error': 'Missing image or conversion type'}, status=400)

        try:
            if conv == 't1-to-t2':
                model = MODELS['t1_to_t2']
                preprocess = standard_preprocess
                postprocess = standard_postprocess
            elif conv == 't2-to-t1':
                model = MODELS['t2_to_t1']
                preprocess = preprocess_for_t2t1
                postprocess = postprocess_for_t2t1
            elif conv == 'pd-to-t2':
                model = MODELS['pd_to_t2']
                preprocess = standard_preprocess
                postprocess = standard_postprocess
            else:
                return Response({'error': 'Invalid conversion type'}, status=400)

            def load_input_image(file, fmt):
                if fmt == 'nii':
                    tf = tempfile.NamedTemporaryFile(suffix='.nii', delete=False)
                    for chunk in file.chunks():
                        tf.write(chunk)
                    tf.close()
                    path = tf.name

                    try:
                        nii = nib.load(path)
                        data = nii.get_fdata()

                        # === NEW: Save axial slices for 3D viewer ===
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        slice_dir = os.path.join("public", "slices", f"slices_{timestamp}")
                        os.makedirs(slice_dir, exist_ok=True)

                        num_slices = data.shape[2]
                        slice_urls = []

                        for i in range(num_slices):
                            slice_path = os.path.join(slice_dir, f"slice_{i}.png")
                            plt.imsave(slice_path, data[:, :, i], cmap='gray')
                            slice_urls.append(f"/media/slices/slices_{timestamp}/slice_{i}.png")

                        # For the rest of processing, still return mid-slice
                        mid_slice = data[:, :, data.shape[2] // 2]
                        norm = (mid_slice - np.min(mid_slice)) / (np.ptp(mid_slice) + 1e-6)
                        pil_img = Image.fromarray((norm * 255).astype(np.uint8)).convert('L')
                        return pil_img, slice_urls
                    finally:
                        os.remove(path)

                else:
                    return Image.open(file).convert('L')

            img, slice_urls = load_input_image(img_f, image_format)
            tensor_input = preprocess(img).unsqueeze(0)

            with torch.no_grad():
                output = model(tensor_input) if conv != 't2-to-t1' else model(tensor_input, label=0)

            # Handle output clamping based on conversion type
            if conv == 't2-to-t1':
                # GeneratorT2T1's Tanh output is effectively [0,1] due to L1 target
                out_clamped = output.squeeze(0).cpu().clamp(0, 1)
            else:
                # Other models output [-1,1]
                out_clamped = output.squeeze(0).cpu().clamp(-1, 1)
                
            pil_out = postprocess(out_clamped)

            img_f.seek(0)
            
            ref_img, _ = load_input_image(img_f, image_format)
            ref_img = ref_img.resize(pil_out.size)

            ref_np = np.array(ref_img, dtype=np.float32) / 255.0
            out_np = np.array(pil_out.convert('L'), dtype=np.float32) / 255.0

            psnr = peak_signal_noise_ratio(ref_np, out_np, data_range=1.0)
            ssim = structural_similarity(ref_np, out_np, data_range=1.0, channel_axis=None, win_size=7)

            ref_rgb = lpips_preprocess_for_lpips(ref_img.convert('RGB')).unsqueeze(0)
            out_rgb = lpips_preprocess_for_lpips(pil_out.convert('RGB')).unsqueeze(0)
            lpips_val = lpips_fn(ref_rgb, out_rgb).item()

            hist, _ = np.histogram((out_np * 255).astype(np.uint8), bins=256, range=(0, 255))

            buf = io.BytesIO()
            pil_out.save(buf, format='JPEG')
            result_b64 = base64.b64encode(buf.getvalue()).decode()

            return Response({
                'result': f'data:image/jpeg;base64,{result_b64}',
                'metrics': {
                    'psnr': psnr,
                    'ssim': ssim,
                    'lpips': lpips_val,
                    'histogram': hist.tolist()
                },
                'sliceUrls': slice_urls if image_format == 'nii' else []  # only if .nii
            })
        
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return Response({'error': f"An error occurred: {str(e)}"}, status=500)

class SegmentImageView(APIView):
    def post(self, request):
        def get_slice(file):
            suffix = '.nii.gz' if file.name.endswith('.nii.gz') else '.nii'
            tf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            for chunk in file.chunks():
                tf.write(chunk)
            tf.close()
            try:
                data = nib.load(tf.name).get_fdata()
                slice_ = data[:, :, data.shape[2] // 2]
                norm = (slice_ - np.min(slice_)) / (np.ptp(slice_) + 1e-6)
                return norm
            finally:
                os.remove(tf.name)

        try:
            t1n  = get_slice(request.FILES['t1n'])
            t1ce = get_slice(request.FILES['t1ce'])
            t2   = get_slice(request.FILES['t2'])

            # Stack into input tensor
            input_tensor = torch.stack([
                torch.tensor(t1n, dtype=torch.float32),
                torch.tensor(t1ce, dtype=torch.float32),
                torch.tensor(t2, dtype=torch.float32)
            ], dim=0).unsqueeze(0)  # [1, 3, H, W]

            input_tensor = F.interpolate(input_tensor, size=(256, 256), mode='bilinear', align_corners=False)
            input_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min())  # [0,1]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_tensor = input_tensor.to(device)

            # Load model
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(BASE_DIR, 'public', 'models', 'cgan_t2f_seg.pt')
            assert os.path.exists(model_path), f"Model not found: {model_path}"

            model = UNet2DGeneratorCheckpointMatch(in_channels=3, out_channels=2)
            
            # Load state dict
            state = torch.load(model_path, map_location='cpu')
            if 'generator' in state:
                state = state['generator']
            elif 'state_dict' in state:
                state = state['state_dict']

            model.load_state_dict(state, strict=False)
            model.to(device)
            model.eval()
            
            with torch.no_grad():
                out = torch.sigmoid(model(input_tensor))
                t2f, seg = out[:, 0:1, :, :], out[:, 1:2, :, :]

            def to_b64(t):
                arr = (t.squeeze().cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(arr)
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

            return Response({
                't2f': to_b64(t2f),
                'seg': to_b64(seg)
            })

        except Exception as e:
            import traceback
            print("Segmentation failed with error:")
            traceback.print_exc()
            return Response({'error': f"{type(e).__name__}: {e}"}, status=500)
        
class IXISegmentView(APIView):
    def post(self, request):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        def to_b64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

        def load_slice(file):
            suffix = '.nii.gz' if file.name.endswith('.nii.gz') else '.nii'
            tf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            for chunk in file.chunks():
                tf.write(chunk)
            tf.close()

            try:
                volume = nib.load(tf.name).get_fdata()
                mid_slice = volume[:, :, volume.shape[2] // 2]
                norm = (mid_slice - np.min(mid_slice)) / (np.ptp(mid_slice) + 1e-6)
                return norm
            finally:
                os.remove(tf.name)

        try:
            print("=== POST /api/ixi-segment/ has been triggered ===")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            t1_slice = load_slice(request.FILES['t1'])
            input_tensor = torch.tensor(t1_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            input_tensor = F.interpolate(input_tensor, size=(256, 256), mode='bilinear', align_corners=False).to(device)

            # === Load model ===
            class IXISimpleUNet(nn.Module):
                def __init__(self, in_channels=1, out_channels=4):
                    super().__init__()
                    self.encoder1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
                                                  nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
                    self.encoder2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
                    self.bottleneck = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                                                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
                    self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
                    self.decoder2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
                    self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
                    self.decoder1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                                                  nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
                    self.final = nn.Conv2d(32, out_channels, 1)

                def forward(self, x):
                    e1 = self.encoder1(x)
                    e2 = self.encoder2(F.max_pool2d(e1, 2))
                    b = self.bottleneck(F.max_pool2d(e2, 2))
                    d2 = self.decoder2(torch.cat([self.up2(b), e2], dim=1))
                    d1 = self.decoder1(torch.cat([self.up1(d2), e1], dim=1))
                    return self.final(d1)

            model = IXISimpleUNet().to(device)
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'ixi_multiclass_model.pt')
            state = torch.load(model_path, map_location=device)
            print("== IXI Model File Debug ==")
            print("Type of loaded object:", type(state))
            if isinstance(state, dict):
                print("Top-level keys:", list(state.keys())[:10])
                for key in list(state.keys())[:5]:
                    print(f"  → {key}")
            else:
                print("Loaded object is not a dict.")

            # Try to extract correct key
            if 'state_dict' in state:
                state = state['state_dict']
            elif 'model' in state:
                state = state['model']
            elif isinstance(state, dict):
                state = state  # Use directly
            else:
                raise RuntimeError("Unrecognized model format")

            model.load_state_dict(state, strict=False)
            model.eval()

            # === Inference ===
            with torch.no_grad():
                output = torch.sigmoid(model(input_tensor))[0]  # shape: [4, H, W]

            # === Overlays ===
            titles = ['CSF', 'Gray Matter', 'White Matter', 'BG/Other']
            colors = [(1, 0, 0, 0.4), (0, 1, 0, 0.4), (0, 0, 1, 0.4), (1, 1, 0, 0.4)]
            overlays = []
            for i in range(4):
                fig, ax = plt.subplots()
                ax.imshow(input_tensor[0, 0].cpu(), cmap='gray')
                ax.imshow(output[i].cpu(), cmap=mcolors.ListedColormap([colors[i]]), alpha=output[i].cpu().numpy())
                ax.axis('off')
                overlays.append(to_b64(fig))
                plt.close(fig)

            # === Hard segmentation map ===
            pred_labels = torch.argmax(output, dim=0).cpu().numpy()
            cmap = mcolors.ListedColormap([(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)])
            bounds = list(range(5))
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            fig1, ax1 = plt.subplots()
            ax1.imshow(pred_labels, cmap=cmap, norm=norm)
            ax1.axis('off')
            hard_seg_b64 = to_b64(fig1)
            plt.close(fig1)

            # === Overlay hard labels ===
            fig2, ax2 = plt.subplots()
            ax2.imshow(input_tensor[0, 0].cpu(), cmap='gray')
            ax2.imshow(pred_labels, cmap=cmap, norm=norm, alpha=0.5)
            ax2.axis('off')
            hard_overlay_b64 = to_b64(fig2)
            plt.close(fig2)

            return Response({
                'overlays': overlays,
                'hardSeg': hard_seg_b64,
                'hardOverlay': hard_overlay_b64
            })
        
        except Exception as e:
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            return Response({'error': f'{type(e).__name__}: {e}'}, status=500)

def extract_slices(nifti_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    img = nib.load(nifti_path)
    data = img.get_fdata()

    num_slices = data.shape[2]  # axial
    for i in range(num_slices):
        plt.imsave(os.path.join(output_dir, f"slice_{i}.png"), data[:, :, i], cmap='gray')
