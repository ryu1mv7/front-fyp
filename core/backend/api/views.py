import io
import base64

import numpy as np
import torch
from PIL import Image
import lpips
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .networks import model_t1_t2, model_pd2t2, preprocess, postprocess

# Initialise LPIPS metric at module level
lpips_fn = lpips.LPIPS(net='alex')  # or 'vgg' for alternative

# Transform to convert PIL image to LPIPS input (normalise to [-1, 1])
lpips_transform = transforms.Compose([
    transforms.Resize((256, 256)),            # Resize to expected input size
    transforms.ToTensor(),                    # Converts to [0,1] tensor
    transforms.Normalize([0.5]*3, [0.5]*3),   # Normalise to [-1,1]
])

class ConvertImageView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        img_f = request.FILES.get('image')
        conv = request.data.get('conversionType')
        if not img_f or not conv:
            return Response({'error': 'Missing image or conversion type'}, status=400)

        # Select the appropriate model
        if conv == 't1-to-t2':
            model = model_t1_t2
        elif conv == 'pd-to-t2':
            model = model_pd2t2
        else:
            return Response({'error': 'Invalid conversion type'}, status=400)

        try:
            # ─── Run the generator ───────────────────────────────────────────
            img = Image.open(img_f).convert('L')
            inp = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                out_t = model(inp)
            pil_out = postprocess(out_t.squeeze(0).cpu().clamp(-1, 1))

            # ─── Compute PSNR & SSIM ─────────────────────────────────────────
            img_f.seek(0)
            orig = Image.open(img_f).convert('L').resize(pil_out.size)
            orig_np = np.asarray(orig, dtype=np.float32) / 255.0
            out_np  = np.asarray(pil_out, dtype=np.float32) / 255.0

            m_psnr = peak_signal_noise_ratio(orig_np, out_np, data_range=1.0)
            m_ssim = structural_similarity(orig_np, out_np, data_range=1.0)

            # ─── Compute LPIPS ────────────────────────────────────────────────
            orig_rgb = orig.convert('RGB')
            gen_rgb  = pil_out.convert('RGB')
            orig_t   = lpips_transform(orig_rgb).unsqueeze(0)
            gen_t    = lpips_transform(gen_rgb).unsqueeze(0)
            lpips_val = lpips_fn(orig_t, gen_t).item()

            # ─── Histogram (optional) ────────────────────────────────────────
            hist, _ = np.histogram(
                (out_np * 255).astype(np.uint8),
                bins=256, range=(0, 255)
            )

            # ─── Encode generated image ───────────────────────────────────────
            buf = io.BytesIO()
            pil_out.save(buf, format='JPEG')
            b64 = base64.b64encode(buf.getvalue()).decode()

            # ─── Return JSON response ─────────────────────────────────────────
            return Response({
                'result': f'data:image/jpeg;base64,{b64}',
                'metrics': {
                    'psnr':  m_psnr,
                    'ssim':  m_ssim,
                    'lpips': lpips_val,
                    'histogram': hist.tolist(),
                }
            })

        except Exception as e:
            return Response({'error': str(e)}, status=500)
