import json
import pandas as pd  # Pandas kütüphanesi şart!
from openai import OpenAI
from tqdm import tqdm

# --- AYARLAR ---
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

INPUT_FILE = "TRSAv1_sample_3k.csv"
OUTPUT_FILE = "kumru2b_sonuclar.json"

# !!! BURAYA DİKKAT !!!
# CSV dosyanı aç, yorumların olduğu sütunun başlığı neyse buraya aynısını yaz.
# Genelde "Review", "text", "content" falan olur.
SUTUN_ADI = "review"

# --- DOSYAYI OKUMA KISMI (Eksik olan yer burasıydı) ---
print(f"CSV okunuyor: {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE)
    # Sütunu listeye çeviriyoruz ki döngü çalışsın
    if SUTUN_ADI not in df.columns:
        print(f"HATA: '{SUTUN_ADI}' sütunu bulunamadı! Mevcut sütunlar: {list(df.columns)}")
        exit()

    yorumlar_listesi = df[SUTUN_ADI].astype(str).tolist()
    print(f"Toplam {len(yorumlar_listesi)} yorum yüklendi.")

except Exception as e:
    print(f"Dosya okuma hatası: {e}")
    exit()

# --- SYSTEM PROMPT ---
system_instruction = """Sen Türkçe metinleri analiz eden, hatasız bir duygu analiz uzmanısın.
Görevin, sana verilen kullanıcı yorumunu analiz etmek ve SADECE aşağıdaki JSON formatında çıktı vermektir.

KULLANILACAK ETİKETLER: "POSITIVE", "NEGATIVE", "NEUTRAL"

KURALLAR (ÇOK ÖNEMLİ):
1. Sadece ve sadece JSON döndür. Başka tek bir kelime bile yazma.
2. Markdown (```json ... ```) kullanma, direkt saf JSON ver.
3. Bu görev projemiz için HAYATİ önem taşıyor. Hata kabul edilemez.

ÖRNEKLER:
Metin: "Yemek harikaydı." -> {"label": "POSITIVE"}
Metin: "Zehirlendim." -> {"label": "NEGATIVE"}
Metin: "Fiyatına göre okey." -> {"label": "NEUTRAL"}
"""

# Sonuçları tutacağımız liste
sonuclar = []

print("Analiz başlıyor...")

# --- DÖNGÜ ---
for yorum in tqdm(yorumlar_listesi):
    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f'Metin: "{yorum}"'}
            ],
            temperature=0.1,
        )

        raw_content = response.choices[0].message.content.strip()
        clean_content = raw_content.replace("```json", "").replace("```", "").strip()

        parsed_json = json.loads(clean_content)

        sonuclar.append({
            "yorum": yorum,
            "analiz": parsed_json
        })

    except json.JSONDecodeError:
        # Hata olsa bile kaydet ki hangisi patladı görelim
        sonuclar.append({
            "yorum": yorum,
            "analiz": {"label": "ERROR", "raw": raw_content}
        })
    except Exception as e:
        print(f"\nBağlantı hatası: {e}")

# --- KAYDETME ---
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(sonuclar, f, ensure_ascii=False, indent=4)

print(f"\nİşlem bitti! Sonuçlar '{OUTPUT_FILE}' dosyasına kaydedildi.")