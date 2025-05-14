import io
import base64
import tempfile
import torch
import numpy as np
import nibabel as nib
from PIL import Image
import lpips
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .networks import (
    model_t1_t2, model_pd2t2, model_brats_t2f_seg, model_mri2ct,
    preprocess, postprocess, BrainSegNet
)

# Initialize LPIPS metric (from general version)
lpips_fn = lpips.LPIPS(net='alex')  # or 'vgg' for alternative
lpips_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# Model mapping (from your version)
MODEL_MAP = {
    't1-to-t2': model_t1_t2,
    'pd-to-t2': model_pd2t2,
    'brats-t2f-seg': model_brats_t2f_seg,
    'mri-to-ct': model_mri2ct,
}

brain_seg_model = None  # Lazy-loaded BrainSegNet (from your version)

class ConvertImageView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        img_f = request.FILES.get('image')
        conv = request.data.get('conversionType')

        if not img_f or not conv:
            return Response({'error': 'Missing image or conversion type'}, status=400)

        global brain_seg_model
        if conv == 'ixi-brain-seg':
            if brain_seg_model is None:
                brain_seg_model = BrainSegNet()
            model = brain_seg_model
        else:
            model = MODEL_MAP.get(conv)

        if not model:
            return Response({'error': 'Invalid conversion type'}, status=400)

        try:
            img = Image.open(img_f).convert('L')
            inp = preprocess(img).unsqueeze(0)

            with torch.no_grad():
                out = model(inp)
                out_t2f = out[0] if isinstance(out, (list, tuple)) else out
                out_seg = out[1] if isinstance(out, (list, tuple)) and len(out) > 1 else out_t2f

            # Postprocess and encode (from your version)
            def encode(tensor):
                image = postprocess(tensor.squeeze(0).cpu().clamp(-1, 1))
                buf = io.BytesIO()
                image.save(buf, format='JPEG')
                return base64.b64encode(buf.getvalue()).decode()

            # Compute metrics (from general version)
            img_f.seek(0)
            orig = Image.open(img_f).convert('L').resize(image.size)
            orig_np = np.asarray(orig, dtype=np.float32) / 255.0
            out_np = np.asarray(image, dtype=np.float32) / 255.0

            m_psnr = peak_signal_noise_ratio(orig_np, out_np, data_range=1.0)
            m_ssim = structural_similarity(orig_np, out_np, data_range=1.0)

            orig_rgb = orig.convert('RGB')
            gen_rgb = image.convert('RGB')
            orig_t = lpips_transform(orig_rgb).unsqueeze(0)
            gen_t = lpips_transform(gen_rgb).unsqueeze(0)
            lpips_val = lpips_fn(orig_t, gen_t).item()

            return Response({
                'result': {
                    't2f': f"data:image/jpeg;base64,{encode(out_t2f)}",
                    'seg': f"data:image/jpeg;base64,{encode(out_seg)}"
                },
                'metrics': {  # From general version
                    'psnr': m_psnr,
                    'ssim': m_ssim,
                    'lpips': lpips_val,
                }
            })
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class ConvertNiftiView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        nii_files = request.FILES.getlist('image')
        conv = request.data.get('conversionType')

        if not nii_files or not conv:
            return Response({'error': 'Missing NIfTI file or conversion type'}, status=400)

        global brain_seg_model
        if conv == 'ixi-brain-seg':
            if brain_seg_model is None:
                brain_seg_model = BrainSegNet()
            model = brain_seg_model
        else:
            model = MODEL_MAP.get(conv)

        if not model:
            return Response({'error': 'Invalid conversion type'}, status=400)

        try:
            if conv == 'ixi-brain-seg':
                # Handle single-modality T1 segmentation
                file = next((f for f in nii_files if 't1' in f.name.lower() and 't1c' not in f.name.lower()), None)
                if not file:
                    return Response({'error': 'No valid T1 NIfTI file found for brain segmentation'}, status=400)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
                    for chunk in file.chunks():
                        tmp.write(chunk)
                    tmp_path = tmp.name

                nii = nib.load(tmp_path)
                data = nii.get_fdata()
                mid_slice = data[:, :, data.shape[2] // 2]

                norm = (mid_slice - np.min(mid_slice)) / (np.max(mid_slice) - np.min(mid_slice) + 1e-8)
                image = Image.fromarray((norm * 255).astype(np.uint8)).convert('L')
                image = image.resize((1024, 1024))  # match model input
                tensor = transforms.ToTensor()(image)
                tensor = transforms.Normalize([0.5], [0.5])(tensor)
                input_tensor = tensor.unsqueeze(0)  # shape: [1, 1, H, W]

                with torch.no_grad():
                    out = model(input_tensor)
                    out_seg = out.argmax(dim=1).squeeze().cpu().numpy()  # [H, W]

                result_img = Image.fromarray((out_seg * 85).astype(np.uint8))  # 0, 85, 170
                buf = io.BytesIO()
                result_img.save(buf, format='JPEG')
                encoded = base64.b64encode(buf.getvalue()).decode()

                return Response({
                    'result': {
                        't2f': None,
                        'seg': f"data:image/jpeg;base64,{encoded}"
                    }
                })

            else:
                # Multi-modal synthesis (e.g. brats-t2f-seg)
                file_map = {f.name.lower(): f for f in nii_files}
                modalities = ['t1n', 't1c', 't2w']
                images = []

                for key in modalities:
                    file = next((f for name, f in file_map.items() if key in name), None)
                    if file:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
                            for chunk in file.chunks():
                                tmp.write(chunk)
                            tmp_path = tmp.name
                        nii = nib.load(tmp_path)
                        data = nii.get_fdata()
                        mid_slice = data[:, :, data.shape[2] // 2]
                        norm = (mid_slice - np.min(mid_slice)) / (np.max(mid_slice) - np.min(mid_slice))
                        image = Image.fromarray((norm * 255).astype(np.uint8)).convert('L')
                        images.append(preprocess(image))

                if not images:
                    return Response({'error': 'No valid input modalities were found'}, status=400)

                input_tensor = torch.cat(images, dim=0).unsqueeze(0)  # [1, C, H, W]

                with torch.no_grad():
                    out = model(input_tensor)
                    out_t2f = out[0] if isinstance(out, (list, tuple)) else out
                    out_seg = out[1] if isinstance(out, (list, tuple)) and len(out) > 1 else out_t2f

                def encode(tensor):
                    image = postprocess(tensor.squeeze(0).cpu().clamp(-1, 1))
                    buf = io.BytesIO()
                    image.save(buf, format='JPEG')
                    return base64.b64encode(buf.getvalue()).decode()

                return Response({
                    'result': {
                        't2f': f"data:image/jpeg;base64,{encode(out_t2f)}",
                        'seg': f"data:image/jpeg;base64,{encode(out_seg)}"
                    }
                })

        except Exception as e:
            return Response({'error': str(e)}, status=500)

class ConvertBatchView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        files = request.FILES.getlist('images')  # 'images' is plural now
        conv = request.data.get('conversionType')

        if not files or not conv:
            return Response({'error': 'Missing images or conversion type'}, status=400)

        model = MODEL_MAP.get(conv)
        if not model:
            return Response({'error': 'Invalid conversion type'}, status=400)

        try:
            outputs = []
            for f in files:
                img = Image.open(f).convert('L')
                inp = preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    out = model(inp)
                    out_tensor = out[0] if isinstance(out, (tuple, list)) else out
                image = postprocess(out_tensor.squeeze(0).cpu().clamp(-1, 1))
                buf = io.BytesIO()
                image.save(buf, format='JPEG')
                encoded = base64.b64encode(buf.getvalue()).decode()
                outputs.append(f"data:image/jpeg;base64,{encoded}")

            return Response({ 'results': outputs })

        except Exception as e:
            return Response({'error': str(e)}, status=500)


class BrainSegView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    def post(self, request):
        file = request.FILES.get('image')
        if not file:
            return Response({'error': 'No NIfTI file provided'}, status=400)
        
        global brain_seg_model
        if brain_seg_model is None:
            brain_seg_model = BrainSegNet()
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
                for chunk in file.chunks():
                    tmp.write(chunk)
                nii = nib.load(tmp.name)
                data = nii.get_fdata()
                slice_ = data[:, :, data.shape[2] // 2]
                norm = (slice_ - np.min(slice_)) / (np.max(slice_) - np.min(slice_))
                image = Image.fromarray((norm * 255).astype(np.uint8)).convert('L')
                tensor = preprocess(image).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    mask = model_ixi_brain_seg(tensor).argmax(dim=1).squeeze().cpu().numpy()
                result_img = Image.fromarray((mask * 85).astype(np.uint8))  # 0, 85, 170
                buf = io.BytesIO()
                result_img.save(buf, format='JPEG')
                return Response({
                    'result': f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
                })
        except Exception as e:
            return Response({'error': str(e)}, status=500)
