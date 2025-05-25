from django import forms

class TahminForm(forms.Form):
    Brüt_Metrekare = forms.FloatField()
    Net_Metrekare = forms.FloatField()
    Binanın_Yaşı = forms.IntegerField()
    Binanın_Kat_Sayısı = forms.IntegerField()
    Eşya_Durumu = forms.IntegerField()  # 0 / 1
    Banyo_Sayısı = forms.IntegerField()
    Oda_Sayısı = forms.IntegerField()
    Bulunduğu_Kat = forms.IntegerField()
    Isıtma_Tipi = forms.IntegerField()  # 0 / 1
    Site_İçerisinde = forms.IntegerField()  # 0 / 1
    yaka = forms.IntegerField()  # 0 = Avrupa, 1 = Anadolu
