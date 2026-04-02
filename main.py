import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy import stats

# 1. VERİ ÇEKME (Son 5 Yıl)
tickers = ['THYAO.IS', 'PGSUS.IS', 'CLEBI.IS']
print("Veriler çekiliyor...")
raw_data = yf.download(tickers, period='5y')['Close']
raw_data.columns = [col.replace('.IS', '') for col in raw_data.columns]
raw_data.dropna(inplace=True)

# Getirileri Hesapla (Returns)
returns = raw_data.pct_change().dropna()

# ---------------------------------------------------------
# ÇİFTE REGRESYON ANALİZİ
# X Ekseni (Bağımsız Değişken): THYAO (Sektör Lideri)
# Y1: PGSUS
# Y2: CLEBI
# ---------------------------------------------------------

# 1. Regresyon: THY -> PEGASUS
slope_pgsus, intercept_pgsus, r_pgsus, p_pgsus, std_pgsus = stats.linregress(returns['THYAO'], returns['PGSUS'])

# 2. Regresyon: THY -> CELEBI
slope_clebi, intercept_clebi, r_clebi, p_clebi, std_clebi = stats.linregress(returns['THYAO'], returns['CLEBI'])

# ---------------------------------------------------------
# RAPORLAMA (Txt Dosyası)
# ---------------------------------------------------------
report = (
    f"--- KARŞILAŞTIRMALI REGRESYON ANALİZİ RAPORU ---\n\n"
    f"BAĞIMSIZ DEĞİŞKEN (X): THYAO (Sektör Lideri)\n\n"
    f"1. PEGASUS (PGSUS) İLİŞKİSİ:\n"
    f"   - Denklem: y = {slope_pgsus:.4f}x + {intercept_pgsus:.4f}\n"
    f"   - Eğim (Beta/Duyarlılık): {slope_pgsus:.4f}\n"
    f"   - R-Kare (Belirleyicilik): {r_pgsus**2:.4f}\n\n"
    f"2. ÇELEBİ (CLEBI) İLİŞKİSİ:\n"
    f"   - Denklem: y = {slope_clebi:.4f}x + {intercept_clebi:.4f}\n"
    f"   - Eğim (Beta/Duyarlılık): {slope_clebi:.4f}\n"
    f"   - R-Kare (Belirleyicilik): {r_clebi**2:.4f}\n\n"
    f"SONUÇ YORUMU:\n"
    f"Hangi hissenin Eğimi (Slope) daha yüksekse, o hisse THY'nin hareketlerine daha sert tepki verir.\n"
)

with open("3_Karsilastirmali_Regresyon_Raporu.txt", "w") as f:
    f.write(report)
print(report)

# ---------------------------------------------------------
# GÖRSELLEŞTİRME (Tek Grafikte İki Çizgi)
# ---------------------------------------------------------
plt.figure(figsize=(12, 7))

# PEGASUS Regresyonu (Kırmızı)
sns.regplot(x=returns['THYAO'], y=returns['PGSUS'],
            scatter_kws={'alpha':0.15, 'color':'red'},
            line_kws={'color':'red', 'label': f'PGSUS (Eğim: {slope_pgsus:.2f})'},
            label='PGSUS Verileri')

# ÇELEBİ Regresyonu (Mavi)
sns.regplot(x=returns['THYAO'], y=returns['CLEBI'],
            scatter_kws={'alpha':0.15, 'color':'blue'},
            line_kws={'color':'blue', 'label': f'CLEBI (Eğim: {slope_clebi:.2f})'},
            label='CLEBI Verileri')

plt.title('THY Hareketlerinin Pegasus ve Çelebi Üzerindeki Etkisi')
plt.xlabel('THY Günlük Getiri (Piyasa Lideri)')
plt.ylabel('Diğer Şirketlerin Getirileri')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("Grafik3_Karsilastirmali_Regresyon.png", dpi=300)
print("Grafik 'Grafik3_Karsilastirmali_Regresyon.png' olarak kaydedildi.")
plt.show()