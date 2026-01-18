import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- DOSYA ADLARI ---
CSV_FILE = "human-labeled-sample-1495.csv"
JSON_FILE = "gemma-3-12b-it_sonuclar.json"


def raporla():
    print("Dosyalar yükleniyor...")

    # 1. Gerçek Verileri (CSV) Oku
    try:
        df_true = pd.read_csv(CSV_FILE, sep=";")

        if 'label' in df_true.columns:
            target_col = 'label'
        elif 'score' in df_true.columns:
            target_col = 'score'
        else:
            print(f"HATA: CSV'de 'label' veya 'score' sütunu bulunamadı! Mevcutlar: {df_true.columns}")
            return

    except Exception as e:
        print(f"HATA: CSV dosyası okunamadı! {e}")
        return

    # 2. Model Sonuçlarını (JSON) Oku
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"HATA: JSON dosyası okunamadı! {e}")
        return

    # 3. Eşleştirme Yap (Index Bazlı)
    y_true_list = []
    y_pred_list = []
    hatali_sayi = 0

    print("Veriler eşleştiriliyor...")

    for item in json_data:
        csv_index = item.get('index')

        if csv_index is not None and csv_index < len(df_true):
            try:
                gercek_deger = str(df_true.iloc[csv_index][target_col]).strip().upper()

                if "error" in item or "analiz" not in item:
                    tahmin = "ERROR"
                    hatali_sayi += 1
                else:
                    tahmin = item['analiz'].get('label', 'UNKNOWN').upper()

                if tahmin in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
                    y_true_list.append(gercek_deger)
                    y_pred_list.append(tahmin)
                else:
                    pass

            except Exception as e:
                print(f"Satır {csv_index} işlenirken hata: {e}")
                continue

    # --- SONUÇLARI HESAPLA ---
    if len(y_true_list) == 0:
        print("HATA: Hiçbir eşleşen veri bulunamadı!")
        return

    print("\n" + "=" * 40)
    print(f"MODEL PERFORMANS RAPORU")
    print(f"Analiz Edilen Toplam Veri: {len(y_true_list)}")
    print("=" * 40)

    # Doğruluk Oranı
    acc = accuracy_score(y_true_list, y_pred_list)
    print(f"GENEL DOĞRULUK (Accuracy): %{acc * 100:.2f}")
    print(f"Hatalı / Format Dışı Sayısı: {hatali_sayi}")
    print("-" * 40)

    target_names = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

    print("\nDETAYLI SINIFLANDIRMA RAPORU:")
    print(classification_report(y_true_list, y_pred_list, labels=target_names, zero_division=0))

    # --- GÖRSEL CONFUSION MATRIX ÇİZİMİ ---
    print("\nKARMAŞIKLIK MATRİSİ ÇİZİLİYOR...")
    cm = confusion_matrix(y_true_list, y_pred_list, labels=target_names)

    # Çizim ayarları
    plt.figure(figsize=(8, 6))  # Pencere boyutu
    sns.set(font_scale=1.2)  # Yazı boyutu

    # Isı haritasını oluştur
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=target_names,
                yticklabels=target_names)

    # Eksen ve Başlıklar
    plt.xlabel('Modelin Tahmini (Predicted)', fontsize=14, labelpad=15)
    plt.ylabel('Gerçek Değer (Actual)', fontsize=14, labelpad=15)
    plt.title('Karmaşıklık Matrisi (Confusion Matrix)', fontsize=16, pad=20)

    # Görseli ekrana bas
    plt.tight_layout()
    plt.show()
    print("=" * 40)


if __name__ == "__main__":
    raporla()