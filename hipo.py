import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats

# 1. VERİ HAZIRLIĞI
tickers = ['THYAO.IS', 'PGSUS.IS', 'CLEBI.IS']
print("Veriler analiz ediliyor...")
raw_data = yf.download(tickers, period='5y')['Close']
raw_data.columns = [col.replace('.IS', '') for col in raw_data.columns]
returns = raw_data.pct_change().dropna()

# 2. P-DEĞERİ MATRİSİ OLUŞTURMA (Otomatik Tablo)
# Boş bir tablo yaratıyoruz
p_matrix = pd.DataFrame(index=returns.columns, columns=returns.columns)
# Boş bir "Karar" tablosu yaratıyoruz (Farklı mı / Benzer mi?)
decision_matrix = pd.DataFrame(index=returns.columns, columns=returns.columns)

for stock_a in returns.columns:
    for stock_b in returns.columns:
        if stock_a == stock_b:
            p_matrix.loc[stock_a, stock_b] = 1.0  # Kendisiyle aynısıdır
            decision_matrix.loc[stock_a, stock_b] = "-"
        else:
            # T-Testi Yap
            t_stat, p_val = stats.ttest_ind(returns[stock_a], returns[stock_b])
            p_matrix.loc[stock_a, stock_b] = p_val

            # Kararı Yaz
            if p_val < 0.05:
                decision_matrix.loc[stock_a, stock_b] = "FARKLIDIR"
            else:
                decision_matrix.loc[stock_a, stock_b] = "BENZERDİR"

# Veri tiplerini float'a çevirelim ki grafik çizebilelim
p_matrix = p_matrix.astype(float)

# 3. SONUÇLARI KAYDETME
print("\n--- ÖZET P-DEĞERİ TABLOSU ---")
print(p_matrix)
print("\n--- KARAR TABLOSU ---")
print(decision_matrix)

p_matrix.to_csv("4_Hipotez_P_Degerleri.csv")
decision_matrix.to_csv("4_Hipotez_Karar_Tablosu.csv")

# 4. GÖRSELLEŞTİRME (HEATMAP - ISI HARİTASI)
plt.figure(figsize=(8, 6))

# Renk skalası: Kırmızı (0'a yakın) = Farklı, Mavi (1'e yakın) = Benzer
sns.heatmap(p_matrix, annot=True, fmt=".3f", cmap="RdYlBu", vmin=0, vmax=1,
            cbar_kws={'label': 'P-Değeri (0.05 Altı = Farklı)'})

plt.title('Hipotez Testi Sonuçları: Kim Kimden Farklı?')
plt.xlabel('Şirket 2')
plt.ylabel('Şirket 1')

plt.savefig("Grafik4_Hipotez_Matrisi.png", dpi=300)
print("\nGrafik 'Grafik4_Hipotez_Matrisi.png' olarak kaydedildi.")
plt.show()