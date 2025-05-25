from django.shortcuts import render
from .forms import TahminForm
import joblib
import numpy as np

def fiyat_tahmin(request):
    tahmini_fiyat = None
    form = TahminForm(request.POST or None)

    if form.is_valid():
        model = joblib.load('analiz/fiyat_modeli.pkl')

        net_m2 = form.cleaned_data['Net_Metrekare']

        girdi = np.array([[
            form.cleaned_data['Brüt_Metrekare'],
            form.cleaned_data['Binanın_Yaşı'],
            form.cleaned_data['Binanın_Kat_Sayısı'],
            form.cleaned_data['Eşya_Durumu'],
            form.cleaned_data['Banyo_Sayısı'],
            net_m2,
            form.cleaned_data['Oda_Sayısı'],
            form.cleaned_data['Bulunduğu_Kat'],
            form.cleaned_data['Isıtma_Tipi'],
            form.cleaned_data['Site_İçerisinde'],
            form.cleaned_data['yaka'],
            0.911,
            178.938
        ]])

        tahmini_fiyat_m2 = np.exp(model.predict(girdi)[0])
        tahmini_fiyat = tahmini_fiyat_m2 * net_m2

    return render(request, 'tahmin.html', {
        'form': form,
        'tahmini_fiyat': tahmini_fiyat
    })
