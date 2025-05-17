from django.urls import path
from .views import ConvertImageView, SegmentImageView

urlpatterns = [
    path('convert/', ConvertImageView.as_view(), name='convert'),
    path('segment/', SegmentImageView.as_view(), name='segment'),  # <--- ADD THIS
]