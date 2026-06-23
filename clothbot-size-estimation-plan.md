# ClothBot — Size Estimation Feature Plan

**Project:** ClothBot (YOLOv8 + Canny Edge clothing classifier)  
**Fitur baru:** Estimasi ukuran pakaian berbasis referensi pelipat pakaian fisik  
**Status:** Planning

---

## 1. Konsep Inti

Pakai **pelipat pakaian sebagai referensi fisik tetap** di frame kamera. Lebar bounding box pakaian dibandingkan terhadap lebar pelipat yang sudah dikalibrasi sekali. Tidak butuh pose estimation, tidak butuh objek eksternal tambahan.

```
rasio = lebar_bbox_pakaian (px) / lebar_pelipat (px)
rasio ≥ 1.0  → XL
rasio ≥ 0.85 → L
rasio ≥ 0.70 → M
rasio < 0.70 → S
```

---

## 2. Asumsi & Constraint

- Kamera **posisi tetap**, tidak berubah selama sesi deteksi
- Pelipat pakaian **selalu dalam frame** saat kalibrasi
- Pakaian diletakkan **di atas atau menutupi** pelipat saat dideteksi
- Threshold rasio bersifat **adjustable** — perlu validasi dengan sampel nyata
- Berlaku untuk kelas: `baju`, `celana`, `baju_lengan_panjang`, `celana_panjang`

---

## 3. Komponen yang Perlu Diimplementasi

### 3.1 Kalibrasi Pelipat (Satu Kali)

Dua opsi — pilih salah satu:

**Opsi A — Manual (hardcode):**  
Ukur lebar pelipat dalam piksel dari frame kamera, simpan ke config.

**Opsi B — Semi-otomatis via Canny (rekomendasi):**  
Deteksi garis vertikal tepi pelipat menggunakan Canny yang sudah ada di project, ambil jarak antar tepi.

```python
# calibration.py
import cv2
import json

def calibrate_folder(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=100, maxLineGap=10)
    # Ambil dua garis vertikal paling dominan → hitung jaraknya
    # Simpan ke config
    folder_width_px = compute_folder_width(lines)
    with open("config.json", "w") as f:
        json.dump({"folder_width_px": folder_width_px}, f)
    return folder_width_px
```

---

### 3.2 Size Estimation Logic

File baru: `size_estimator.py`

```python
# size_estimator.py
import json

SIZE_THRESHOLDS = {
    "XL": 1.0,
    "L":  0.85,
    "M":  0.70,
    "S":  0.0
}

def load_folder_width():
    with open("config.json") as f:
        return json.load(f)["folder_width_px"]

def estimate_size(bbox_width_px: float, folder_width_px: float) -> str:
    ratio = bbox_width_px / folder_width_px
    for size, threshold in SIZE_THRESHOLDS.items():
        if ratio >= threshold:
            return size
    return "S"
```

Threshold `SIZE_THRESHOLDS` bisa diubah dari config tanpa edit kode.

---

### 3.3 Integrasi ke Pipeline YOLOv8

Di file inference utama lo, tambahkan size estimation setelah deteksi:

```python
# inference.py (modifikasi)
from size_estimator import estimate_size, load_folder_width

folder_width = load_folder_width()

results = yolo_model(frame)
for box in results[0].boxes:
    label = yolo_model.names[int(box.cls)]
    if label == "null":
        continue

    x1, y1, x2, y2 = box.xyxy[0]
    bbox_width = x2 - x1

    size = estimate_size(float(bbox_width), folder_width)
    output_label = f"{label} - {size}"

    # Render label ke frame
    cv2.putText(frame, output_label, (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
```

---

### 3.4 Visualisasi di Frame (Overlay)

Tambahkan visual helper agar mudah di-debug dan presentasi:

- **Garis vertikal** menandai batas kiri-kanan pelipat (warna biru)
- **Bounding box pakaian** dengan warna berbeda per ukuran
- **Label** format: `baju - L (rasio: 0.87)`
- **Mode debug toggle** — bisa dimatikan saat demo final

```python
# visualization.py
COLOR_MAP = {"XL": (0,0,255), "L": (0,165,255), "M": (0,255,0), "S": (255,0,0)}

def draw_folder_reference(frame, folder_x1, folder_x2):
    h = frame.shape[0]
    cv2.line(frame, (folder_x1, 0), (folder_x1, h), (255, 200, 0), 2)
    cv2.line(frame, (folder_x2, 0), (folder_x2, h), (255, 200, 0), 2)

def draw_result(frame, bbox, label, size, ratio, debug=False):
    x1, y1, x2, y2 = bbox
    color = COLOR_MAP.get(size, (255,255,255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} - {size}"
    if debug:
        text += f" ({ratio:.2f})"
    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
```

---

## 4. Struktur File

```
clothbot/
├── config.json              ← hasil kalibrasi (auto-generate)
├── calibration.py           ← NEW: deteksi lebar pelipat
├── size_estimator.py        ← NEW: logika S/M/L/XL
├── visualization.py         ← NEW: overlay helper
├── inference.py             ← MODIFIED: tambah size estimation
└── ...file existing lainnya
```

---

## 5. Alur Penggunaan

```
[Pertama kali / re-kalibrasi]
  1. Jalankan calibration.py dengan pelipat kosong di frame
  2. Verifikasi garis biru tepat di tepi pelipat
  3. Simpan → config.json

[Setiap sesi deteksi]
  1. Letakkan pakaian di atas pelipat
  2. Jalankan inference.py
  3. Output: jenis + ukuran muncul di frame
```

---

## 6. Validasi & Tuning Threshold

Langkah wajib sebelum finalisasi:

| Langkah | Aksi |
|---------|------|
| 1 | Kumpulkan minimal 3 baju per ukuran (S, M, L, XL) |
| 2 | Jalankan inference dengan `debug=True` |
| 3 | Catat rasio aktual tiap pakaian |
| 4 | Sesuaikan `SIZE_THRESHOLDS` di `size_estimator.py` |
| 5 | Re-test hingga akurasi ≥ 90% pada sampel |

> Threshold default (0.70 / 0.85 / 1.0) adalah estimasi awal — **wajib dikalibrasi** dengan pakaian asli lo.

---

## 7. Edge Cases & Penanganan

| Kondisi | Penanganan |
|---------|------------|
| Pakaian ditekuk / tidak terbuka penuh | Tambahkan minimum confidence threshold — hanya proses jika `box.conf ≥ 0.7` |
| Pelipat tidak terdeteksi saat kalibrasi | Fallback ke input manual width via CLI argument |
| Dua pakaian dalam satu frame | Proses tiap bounding box secara independen |
| Kamera bergeser setelah kalibrasi | Paksa re-kalibrasi — tambahkan flag warning jika folder_x1/x2 tidak terdeteksi |

---

## 8. Prioritas Implementasi

```
[1] size_estimator.py       → logika utama, paling kritis
[2] Integrasi inference.py  → supaya bisa langsung ditest
[3] calibration.py          → mulai manual dulu, otomasi belakangan
[4] visualization.py        → polish, bisa dikerjakan terakhir
```

---

*Plan ini fokus pada implementasi minimal yang fungsional terlebih dahulu, lalu iterasi.*
