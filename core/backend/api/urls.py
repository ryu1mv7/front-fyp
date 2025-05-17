from django.urls import path
from .views import ConvertImageView

urlpatterns = [
    path('convert/', ConvertImageView.as_view(), name='convert'),
    path('api/segment/', SegmentationView.as_view()),
]
