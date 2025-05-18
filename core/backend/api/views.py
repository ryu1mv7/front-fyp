import io
import base64
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
                        mid_slice = data[:, :, data.shape[2] // 2] if data.ndim == 3 else data
                        norm = (mid_slice - np.min(mid_slice)) / (np.ptp(mid_slice) + 1e-6)
                        return Image.fromarray((norm * 255).astype(np.uint8)).convert('L')
                    finally:
                        os.remove(path)
                else:
                    return Image.open(file).convert('L')

            img = load_input_image(img_f, image_format)
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
            ref_img = load_input_image(img_f, image_format).resize(pil_out.size)
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
                }
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