from django.contrib import admin
from django.urls import path, include   # ← add include here

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),  # ← this wires in your app’s URLs
]
