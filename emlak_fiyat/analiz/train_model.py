import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('guncellenmis_veri_seti.xlsx')

X = df[['Brüt_Metrekare', 'Binanın_Yaşı', 'Binanın_Kat_Sayısı', 'Eşya_Durumu',
        'Banyo_Sayısı', 'Net_Metrekare', 'Oda_Sayısı', 'Bulunduğu_Kat',
        'Isıtma_Tipi', 'Site_İçerisinde', 'yaka', 'Yaşam_endeksi', 'Nüfus']]

y = np.log(df['Fiyat_m2'])  # log dönüşümü

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=8, random_seed=42, verbose=False)
model.fit(X_train, y_train)


y_pred_log = model.predict(X_test)
y_pred = np.exp(y_pred_log)
y_true = np.exp(y_test)


mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("Model Performans Özeti")
print("-" * 30)
print(f"MAE (Fiyat/m²): {mae:.2f} TL")
print(f"R² Skoru: {r2:.3f}")


joblib.dump(model, 'analiz/fiyat_modeli.pkl')

#Korelasyon Matrisi
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Korelasyon Matrisi')
plt.tight_layout()
plt.savefig("static/korelasyon_matrisi.png")
plt.show()

#Gerçek vs Tahmin Fiyatlar
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel("Gerçek Fiyat (m²)")
plt.ylabel("Tahmin Edilen Fiyat (m²)")
plt.title("Gerçek vs Tahmin Edilen Fiyatlar")
plt.tight_layout()
plt.savefig("static/gercek_vs_tahmin.png")
plt.show()

#Özellik Önem Grafiği
importances = model.get_feature_importance()
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Özellik Önem Grafiği")
plt.xlabel("Önem Derecesi")
plt.tight_layout()
plt.savefig("static/ozellik_onemi.png")
plt.show()

#Tahmin Hatalarının Dağılımı (Histogram)
residuals = y_true - y_pred

plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Tahmin Hatalarının (Residuals) Dağılımı")
plt.xlabel("Gerçek - Tahmin")
plt.ylabel("Frekans")
plt.tight_layout()
plt.savefig("static/residuals_hist.png")
plt.show()

#Gerçek Fiyata Göre Hata Dağılımı (Scatter)
plt.figure(figsize=(8, 6))
plt.scatter(y_true, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Gerçek Fiyat (m²)")
plt.ylabel("Tahmin Hatası")
plt.title("Gerçek Fiyata Göre Hata Dağılımı")
plt.tight_layout()
plt.savefig("static/residuals_vs_true.png")
plt.show()

#Değişkenlerin Dağılımı
features_to_plot = ['Net_Metrekare', 'Brüt_Metrekare', 'Oda_Sayısı', 'Fiyat_m2']
for feature in features_to_plot:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[feature], bins=30, kde=True)
    plt.title(f"{feature} Dağılımı")
    plt.tight_layout()
    plt.savefig(f"static/dagilim_{feature}.png")
    plt.show()
