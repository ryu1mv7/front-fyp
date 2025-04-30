import io
import base64
import torch              
from PIL import Image
from rest_framework.views import APIView


import io
import base64
from PIL import Image
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser

from .networks import model_t1_t2, model_pd2t2, preprocess, postprocess

class ConvertImageView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        img_f = request.FILES.get('image')
        conv = request.data.get('conversionType')
        if not img_f or not conv:
            return Response({'error': 'Missing image or conversion type'}, status=400)

        if conv == 't1-to-t2':
            model = model_t1_t2
        elif conv == 'pd-to-t2':
            model = model_pd2t2
        else:
            return Response({'error': 'Invalid conversion type'}, status=400)

        try:
            img = Image.open(img_f).convert('L')
            inp = preprocess(img).unsqueeze(0)

            with torch.no_grad():
                out_t = model(inp)

            pil_out = postprocess(out_t.squeeze(0).cpu().clamp(-1,1))

            buf  = io.BytesIO()
            pil_out.save(buf, format='JPEG')
            data = base64.b64encode(buf.getvalue()).decode()
            return Response({'result': f'data:image/jpeg;base64,{data}'})

        except Exception as e:
            return Response({'error': str(e)}, status=500)
