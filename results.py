import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- DOSYA ADLARI ---
# (Eğer isimleri değiştirdiysen burayı düzelt)
CSV_FILE = "TRSAv1_sample_3k.csv"
JSON_FILE = "qwen3-14b_sonuclar.json"


def raporla():
    print("Dosyalar yükleniyor...")

    # 1. Gerçek Verileri (CSV) Oku
    try:
        df_true = pd.read_csv(CSV_FILE)
        # CSV'deki etiketler 'Positive', 'Negative' gibi olabilir.
        # Hepsini BÜYÜK HARFE çevirelim ki 'Positive' == 'POSITIVE' olsun.
        y_true = df_true['score'].str.upper().tolist()
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

    # 3. Model Tahminlerini Listeye Çevir
    y_pred = []
    hatali_sayi = 0

    # JSON ve CSV satır satır aynı sırada ilerlediği için direkt index ile eşleşir
    for item in json_data:
        try:
            # Modelin cevabını al (Eğer hata varsa 'ERROR' basarız)
            if "error" in item or "analiz" not in item:
                tahmin = "ERROR"
                hatali_sayi += 1
            else:
                # {"label": "POSITIVE"} kısmından label'ı çekiyoruz
                tahmin = item['analiz'].get('label', 'UNKNOWN').upper()

            y_pred.append(tahmin)

        except:
            y_pred.append("ERROR")
            hatali_sayi += 1

    # Uzunluk Kontrolü (CSV ile JSON satır sayısı tutuyor mu?)
    if len(y_true) != len(y_pred):
        print(f"UYARI: CSV satır sayısı ({len(y_true)}) ile JSON sonucu ({len(y_pred)}) eşit değil!")
        # Eşit değilse en kısa olana göre keselim (mecburen)
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    # --- SONUÇLARI HESAPLA ---
    print("\n" + "=" * 40)
    print(f"MODEL PERFORMANS RAPORU (GPT-OSS-20B)")
    print("=" * 40)

    # Doğruluk Oranı
    acc = accuracy_score(y_true, y_pred)
    print(f"GENEL DOĞRULUK (Accuracy): %{acc * 100:.2f}")
    print(f"Format Hatası / Bozuk JSON Sayısı: {hatali_sayi}")
    print("-" * 40)

    # Detaylı Rapor (Precision, Recall, F1)
    # labels parametresi ile sadece beklediğimiz sınıfları raporluyoruz (ERROR'ları dışlıyoruz)
    target_names = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    print("\nDETAYLI SINIFLANDIRMA RAPORU:")
    print(classification_report(y_true, y_pred, labels=target_names, zero_division=0))

    # Karmaşıklık Matrisi (Model neleri karıştırıyor?)
    print("\nKARMAŞIKLIK MATRİSİ (Confusion Matrix):")
    # Satırlar: Gerçekler, Sütunlar: Tahminler
    cm = confusion_matrix(y_true, y_pred, labels=target_names)
    cm_df = pd.DataFrame(cm, index=[f"Gerçek {x}" for x in target_names],
                         columns=[f"Tahmin {x}" for x in target_names])
    print(cm_df)
    print("=" * 40)


if __name__ == "__main__":
    raporla()