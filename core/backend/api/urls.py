from django.urls import path
from .views import ConvertImageView, ConvertNiftiView, ConvertBatchView, BrainSegView

urlpatterns = [
    path('convert/', ConvertImageView.as_view(), name='convert'),
    path('convert_nii/', ConvertNiftiView.as_view(), name='convert_nii'),
    path('convert_batch/', ConvertBatchView.as_view(), name='convert_batch'),
    path('brain_seg/', BrainSegView.as_view(), name='brain_seg'),
]