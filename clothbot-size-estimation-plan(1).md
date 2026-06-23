# ClothBot — Size Estimation Feature Plan

**Project:** ClothBot (YOLOv8 + Canny Edge clothing classifier)  
**Fitur baru:** Estimasi ukuran pakaian berbasis referensi pelipat pakaian fisik  
**Status:** Planning  
**Versi:** 1.1

---

## 1. Konsep Inti

Pakai **pelipat pakaian sebagai referensi fisik tetap** di frame kamera. Lebar bounding box pakaian dikonversi ke **satuan cm** menggunakan skala dari hasil kalibrasi, lalu dimap ke tabel ukuran standar.

**Referensi fisik:**
- Area total = panel hitam (pelipat) + kardus ekstensi bawah
- **Lebar total: 90 cm** → acuan utama estimasi ukuran
- **Tinggi total: 56 cm** → opsional, acuan panjang pakaian

```
skala        = 90 cm / lebar_referensi_px   (hasil kalibrasi)
lebar_baju   = bbox_width_px × skala        (dalam cm)
→ map ke tabel ukuran
```

---

## 2. Asumsi & Constraint

- Kamera **posisi tetap dan tegak lurus** (top-down), tidak berubah setelah kalibrasi
- Area referensi (pelipat + kardus) **selalu dalam frame** saat kalibrasi maupun deteksi
- Pakaian diletakkan **flat dan terbuka penuh** di atas area referensi
- Kalibrasi dilakukan **sekali**, ulang hanya jika kamera bergeser
- Area referensi terdiri dari **dua material berbeda** (panel hitam + kardus coklat) — tidak bisa diandalkan untuk auto-detect Canny, wajib kalibrasi manual klik
- Berlaku untuk kelas: `baju`, `celana`, `baju_lengan_panjang`, `celana_panjang`

---

## 3. Alur Sistem

```
[SETUP — satu kali]
  Jalankan calibration.py
      → Tampilkan live feed kamera
      → User klik titik kiri & kanan area referensi
      → Hitung lebar_px, skala cm/px
      → Simpan ke config.json

[SETIAP SESI DETEKSI]
  Jalankan inference.py
      → Load config.json
      → YOLOv8 deteksi jenis pakaian
      → Konversi bbox_width_px → cm via skala
      → Map ke S / M / L / XL / XXL
      → Render label + overlay di frame
```

---

## 4. Komponen yang Perlu Diimplementasi

### 4.1 `calibration.py` — Kalibrasi Manual Klik (WAJIB PERTAMA)

User klik dua titik: tepi kiri dan tepi kanan area total referensi (termasuk kardus).  
Script otomatis hitung piksel dan skala, lalu simpan ke `config.json`.

```python
import cv2, json

FOLDER_WIDTH_CM = 90.0
points = []

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append(x)
        print(f"Titik {len(points)}: x={x}px")

cap = cv2.VideoCapture(0)
cv2.namedWindow("Kalibrasi — Klik tepi KIRI lalu tepi KANAN area referensi")
cv2.setMouseCallback("Kalibrasi — Klik tepi KIRI lalu tepi KANAN area referensi", on_click)

while True:
    ret, frame = cap.read()
    # Tampilkan titik yang sudah diklik
    for px in points:
        cv2.line(frame, (px, 0), (px, frame.shape[0]), (0, 255, 255), 2)

    if len(points) == 2:
        lebar_px = abs(points[1] - points[0])
        skala = FOLDER_WIDTH_CM / lebar_px
        cv2.putText(frame, f"Lebar: {lebar_px}px = {FOLDER_WIDTH_CM}cm | Skala: {skala:.4f} cm/px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, "Tekan S untuk simpan, R untuk reset",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

    cv2.imshow("Kalibrasi — Klik tepi KIRI lalu tepi KANAN area referensi", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and len(points) == 2:
        lebar_px = abs(points[1] - points[0])
        config = {
            "folder_width_cm": FOLDER_WIDTH_CM,
            "folder_width_px": lebar_px,
            "folder_x1": min(points),
            "folder_x2": max(points),
            "scale_cm_per_px": round(FOLDER_WIDTH_CM / lebar_px, 4)
        }
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"✓ Kalibrasi disimpan: {config}")
        break
    elif key == ord('r'):
        points.clear()
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

### 4.2 `size_estimator.py` — Konversi cm → Ukuran

Konversi dari piksel ke cm menggunakan skala, lalu map ke ukuran berdasarkan lebar baju standar Indonesia.

```python
import json

# Threshold dalam CM — sesuaikan setelah validasi dengan baju asli
SIZE_MAP = [
    ("XXL", 68),
    ("XL",  59),
    ("L",   54),
    ("M",   49),
    ("S",   0),
]

def load_config():
    with open("config.json") as f:
        return json.load(f)

def estimate_size(bbox_width_px: float, scale_cm_per_px: float) -> tuple[str, float]:
    lebar_cm = bbox_width_px * scale_cm_per_px
    for size, min_cm in SIZE_MAP:
        if lebar_cm >= min_cm:
            return size, round(lebar_cm, 1)
    return "S", round(lebar_cm, 1)
```

> Threshold `SIZE_MAP` dalam **cm asli** — lebih intuitif untuk di-tune dibanding rasio.

---

### 4.3 Modifikasi `inference.py` — Integrasi Size Estimation

```python
from size_estimator import load_config, estimate_size

config = load_config()
scale = config["scale_cm_per_px"]
folder_x1 = config["folder_x1"]
folder_x2 = config["folder_x2"]

results = yolo_model(frame)
for box in results[0].boxes:
    label = yolo_model.names[int(box.cls)]
    if label == "null" or box.conf < 0.7:
        continue

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    bbox_width = x2 - x1

    size, lebar_cm = estimate_size(float(bbox_width), scale)
    output_label = f"{label} - {size} (~{lebar_cm}cm)"

    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_MAP[size], 2)
    cv2.putText(frame, output_label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_MAP[size], 2)
```

---

### 4.4 `visualization.py` — Overlay Helper

```python
COLOR_MAP = {
    "XXL": (0, 0, 255),
    "XL":  (0, 80, 255),
    "L":   (0, 165, 255),
    "M":   (0, 255, 0),
    "S":   (255, 180, 0)
}

def draw_reference_lines(frame, x1, x2):
    """Gambar garis batas kiri-kanan area referensi kalibrasi."""
    h = frame.shape[0]
    cv2.line(frame, (x1, 0), (x1, h), (255, 220, 0), 2)
    cv2.line(frame, (x2, 0), (x2, h), (255, 220, 0), 2)
    cv2.putText(frame, "REF", (x1 + 4, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1)

def draw_result(frame, bbox, label, size, lebar_cm, debug=False):
    x1, y1, x2, y2 = bbox
    color = COLOR_MAP.get(size, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} - {size}"
    if debug:
        text += f" ({lebar_cm}cm)"
    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
```

---

## 5. Struktur File

```
clothbot/
├── config.json              ← AUTO-GENERATED saat kalibrasi
├── calibration.py           ← NEW: kalibrasi manual klik (jalankan pertama)
├── size_estimator.py        ← NEW: konversi px → cm → ukuran
├── visualization.py         ← NEW: overlay helper
├── inference.py             ← MODIFIED: integrasi size estimation
└── ...file existing lainnya
```

---

## 6. config.json — Format Output Kalibrasi

```json
{
  "folder_width_cm": 90.0,
  "folder_width_px": 720,
  "folder_x1": 110,
  "folder_x2": 830,
  "scale_cm_per_px": 0.1250
}
```

Nilai `folder_width_px` akan berbeda tergantung resolusi kamera dan jarak mounting.

---

## 7. Tabel Ukuran Referensi (Default)

Berdasarkan standar pakaian Asia/Indonesia. **Wajib divalidasi** dengan baju asli.

| Ukuran | Lebar baju (cm) | Rasio vs 90 cm |
|--------|----------------|----------------|
| S      | < 49 cm        | < 0.54         |
| M      | 49 – 53 cm     | 0.54 – 0.59    |
| L      | 54 – 58 cm     | 0.60 – 0.64    |
| XL     | 59 – 67 cm     | 0.65 – 0.74    |
| XXL    | > 67 cm        | > 0.75         |

---

## 8. Validasi & Tuning Threshold

| Langkah | Aksi |
|---------|------|
| 1 | Siapkan minimal 2–3 baju per ukuran (S, M, L, XL, XXL) |
| 2 | Jalankan inference dengan `debug=True` |
| 3 | Catat `lebar_cm` aktual tiap baju dari output |
| 4 | Sesuaikan threshold `SIZE_MAP` di `size_estimator.py` |
| 5 | Re-test hingga akurasi ≥ 90% pada semua sampel |

---

## 9. Edge Cases & Penanganan

| Kondisi | Penanganan |
|---------|------------|
| `config.json` belum ada | Tampilkan error jelas: *"Jalankan calibration.py terlebih dahulu"* |
| Pakaian tidak terbuka penuh | Skip jika `box.conf < 0.7` |
| Dua pakaian dalam satu frame | Proses tiap bounding box secara independen |
| Kamera bergeser setelah kalibrasi | Jalankan ulang `calibration.py`, timpa `config.json` |
| Estimasi ukuran tidak akurat | Tuning `SIZE_MAP` — jangan ubah skala kalibrasi |

---

## 10. Prioritas Implementasi

```
[1] calibration.py           → WAJIB pertama, tanpa ini sistem tidak bisa jalan
[2] size_estimator.py        → logika utama konversi px → cm → ukuran
[3] Modifikasi inference.py  → integrasi ke pipeline yang sudah ada
[4] visualization.py         → polish overlay, kerjakan terakhir
```

---

*Kalibrasi adalah fondasi sistem ini — lakukan sekali dengan benar, deteksi berjalan akurat selama posisi kamera tidak berubah.*
