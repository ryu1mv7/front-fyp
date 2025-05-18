from django.urls import path
from .views import ConvertImageView, SegmentImageView, IXISegmentView

urlpatterns = [
    path('convert/', ConvertImageView.as_view(), name='convert'),
    path('segment/', SegmentImageView.as_view(), name='segment'),
    path('ixi-segment/', IXISegmentView.as_view(), name='ixi-segment'), 
]