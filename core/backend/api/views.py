import io
import base64
import numpy as np
import torch
from PIL import Image
import lpips
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import tempfile
import os # Added for os.remove

from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

import nibabel as nib

from .networks import (
    model_t1_t2,
    model_pd2t2,
    model_t2_t1,
    # preprocess and postprocess are now context-dependent
)

# --- Initialise LPIPS metric at module level ---
lpips_fn = lpips.LPIPS(net='alex')

# --- Transforms ---
# Standard preprocess for models expecting [-1,1] input (T1->T2, PD->T2)
standard_preprocess = transforms.Compose([
    transforms.Resize((256,256), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]), # To [-1, 1]
])

# Preprocess for T2->T1 model (trained with [0,1] input)
preprocess_for_t2t1 = transforms.Compose([
    transforms.Resize((256,256), interpolation=Image.BICUBIC),
    transforms.ToTensor(), # To [0, 1]
])

# Standard postprocess for models outputting [-1,1] (T1->T2, PD->T2)
standard_postprocess = transforms.Compose([
    transforms.Normalize([-1], [2]), # From [-1, 1] to [0, 1]
    transforms.ToPILImage()
])

# Postprocess for T2->T1 model (Tanh output effectively in [0,1] due to L1 target)
postprocess_for_t2t1 = transforms.Compose([
    # Input tensor is already effectively in [0,1] (clamped from Tanh output)
    transforms.ToPILImage()
])

# Transform for LPIPS (normalize to [-1, 1], 3 channels)
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

        current_model = None
        active_preprocess = None
        active_postprocess = None

        if conv == 't1-to-t2':
            current_model = model_t1_t2
            active_preprocess = standard_preprocess
            active_postprocess = standard_postprocess
        # This model accepts t2w inputs
        elif conv == 't2-to-t1':
            current_model = model_t2_t1
            active_preprocess = preprocess_for_t2t1 # Use [0,1] input
            active_postprocess = postprocess_for_t2t1  # Expect effective [0,1] output
        elif conv == 'pd-to-t2':
            current_model = model_pd2t2
            active_preprocess = standard_preprocess
            active_postprocess = standard_postprocess
        else:
            return Response({'error': 'Invalid conversion type'}, status=400)

        try:
            def load_input_image(uploaded_file, fmt):
                # ... (load_input_image function remains the same as your previous version, ensure np.min/np.max handling for NIfTI)
                if fmt == 'nii':
                    path_to_delete = None
                    if hasattr(uploaded_file, 'temporary_file_path'):
                        path = uploaded_file.temporary_file_path()
                    else:
                        suffix = '.nii'
                        tf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                        path_to_delete = tf.name # Mark for deletion
                        for chunk in uploaded_file.chunks():
                            tf.write(chunk)
                        tf.close() # Close the file before nibabel opens it
                        path = path_to_delete
                    
                    try:
                        nii = nib.load(path)
                        data = nii.get_fdata()
                        if data.ndim < 3:
                            if data.ndim == 2:
                                slice2 = data
                            else:
                                raise ValueError("NIfTI data has less than 2 dimensions.")
                        else:
                            mid = data.shape[2] // 2
                            slice2 = data[:, :, mid]
                        
                        # Correct normalization for the slice
                        slice_min = np.min(slice2)
                        slice_max = np.max(slice2)
                        if slice_max == slice_min: # Handle uniform slice
                            norm = np.zeros_like(slice2, dtype=np.float32)
                        else:
                            norm = (slice2 - slice_min) / (slice_max - slice_min)
                        
                        arr = (norm * 255).astype(np.uint8)
                        return Image.fromarray(arr).convert('L')
                    finally:
                        if path_to_delete: # Clean up temp file if we created it
                             os.remove(path_to_delete)
                else:
                    return Image.open(uploaded_file).convert('L')

            img = load_input_image(img_f, image_format)
            inp = active_preprocess(img).unsqueeze(0)

            with torch.no_grad():
                if conv == 't2-to-t1':
                    label_for_t2_to_t1 = 0 # Use 1 for BraTS-style T1 output
                    out_t = current_model(inp, label=label_for_t2_to_t1)
                else:
                    out_t = current_model(inp)
            
            # Clamp and postprocess
            if conv == 't2-to-t1':
                # GeneratorT2T1's Tanh output is effectively [0,1] due to L1 target.
                # Clamp ensures it, then ToPILImage handles the [0,1] tensor.
                pil_out = active_postprocess(out_t.squeeze(0).cpu().clamp(0, 1))
            else:
                # Other models output [-1,1], standard_postprocess maps [-1,1] to [0,1] then PIL.
                pil_out = active_postprocess(out_t.squeeze(0).cpu().clamp(-1, 1))


            # --- Metrics ---
            img_f.seek(0)
            orig_pil_for_metrics = load_input_image(img_f, image_format).resize(pil_out.size)
            orig_np = np.asarray(orig_pil_for_metrics, dtype=np.float32) / 255.0 # Range [0,1]
            
            # Generated image for metrics also needs to be in [0,1] float
            # pil_out is already in the correct display range (0-255 after ToPILImage)
            # So, convert pil_out back to numpy array [0,1] for metrics
            out_np_for_display_and_metrics = np.asarray(pil_out.convert('L'), dtype=np.float32) / 255.0


            m_psnr = peak_signal_noise_ratio(orig_np, out_np_for_display_and_metrics, data_range=1.0)
            m_ssim = structural_similarity(orig_np, out_np_for_display_and_metrics, data_range=1.0, channel_axis=None, win_size=7) # Assuming grayscale

            # LPIPS
            orig_rgb_pil = orig_pil_for_metrics.convert('RGB')
            gen_rgb_pil  = pil_out.convert('RGB') # pil_out is the final displayable PIL image

            orig_t_lpips = lpips_preprocess_for_lpips(orig_rgb_pil).unsqueeze(0)
            gen_t_lpips  = lpips_preprocess_for_lpips(gen_rgb_pil).unsqueeze(0)
            lpips_val = lpips_fn(orig_t_lpips, gen_t_lpips).item()

            # Histogram
            hist, _ = np.histogram(
                (out_np_for_display_and_metrics * 255).astype(np.uint8), # Use the [0,1] float array
                bins=256, range=(0, 255)
            )

            # --- Encode generated image for display ---
            buf = io.BytesIO()
            pil_out.save(buf, format='JPEG') # pil_out is already the correct display image
            b64 = base64.b64encode(buf.getvalue()).decode()

            return Response({
                'result': f'data:image/jpeg;base64,{b64}',
                'metrics': {
                    'psnr':      m_psnr,
                    'ssim':      m_ssim,
                    'lpips':     lpips_val,
                    'histogram': hist.tolist(),
                }
            })

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return Response({'error': f"An error occurred: {str(e)}"}, status=500)
