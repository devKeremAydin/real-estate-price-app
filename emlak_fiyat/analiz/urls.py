from django.urls import path
from .views import fiyat_tahmin

urlpatterns = [
    path('', fiyat_tahmin, name='fiyat_tahmin'),
]
