# ClothBot — Panduan Setup & Menjalankan

## Struktur Proyek

```
gce/
├── app.py                  # FastAPI server (inference + fold + size estimation)
├── size_estimator.py       # Logika estimasi ukuran pakaian
├── calibration.py          # Kalibrasi papan pelipat (skala & perspektif)
├── visualization.py        # Overlay anotasi gambar
├── mqtt_publisher.py       # Publisher perintah servo via MQTT
├── servo_map.py            # Pemetaan label → sequence servo
├── best.onnx               # Model YOLOv8-cls (tidak di-commit jika besar)
├── labels.txt              # Daftar kelas YOLO (1 per baris)
├── config.json             # Hasil kalibrasi (auto-generated)
├── .env                    # Environment variables (buat manual, lihat contoh)
├── requirements.txt        # Python dependencies
├── run.sh                  # Script run server
└── clothbot_esp32/         # Firmware Arduino untuk ESP32 servo controller
    ├── clothbot_esp32.ino
    └── secrets.h           # Isi WiFi + MQTT credentials (jangan di-commit!)
```

---

## Prasyarat

| Kebutuhan | Versi minimum | Cek |
|-----------|--------------|-----|
| Python    | 3.10+        | `python --version` |
| pip       | terbaru      | `pip --version` |
| Camera    | USB/MIPI     | untuk kalibrasi live |

> **Python 3.10+** wajib — kode menggunakan sintaks `str \| None` (PEP 604).

---

## Setup Pertama Kali (Step by Step)

### 1. Masuk ke direktori proyek

```bash
cd gce
```

### 2. Buat virtual environment

```bash
python -m venv .venv
```

### 3. Aktifkan virtual environment

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Prompt terminal akan berubah jadi `(.venv) ...` jika berhasil.

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Proses ini bisa memakan waktu beberapa menit karena `onnxruntime` dan `opencv` cukup besar.

### 5. Buat file `.env`

Buat file `.env` di dalam folder `gce/`:

```bash
# Salin contoh di bawah ini ke file .env
```

Isi minimal `.env`:

```env
MQTT_ENABLED=false
FOLD_ENABLED=true
SIZE_ESTIMATION_ENABLED=true
SIZE_DEBUG_MODE=false
CONF_THRESHOLD=0.75
```

Jika menggunakan MQTT (untuk servo ESP32):

```env
MQTT_ENABLED=true
MQTT_HOST=localhost          # atau IP broker MQTT kamu
MQTT_PORT=1883
MQTT_USER=clothbot
MQTT_PASSWORD=ganti_ini
MQTT_TOPIC=clothbot/servo/command
FOLD_ENABLED=true
SIZE_ESTIMATION_ENABLED=true
SIZE_DEBUG_MODE=false
CONF_THRESHOLD=0.75
```

### 6. Pastikan file model tersedia

```bash
ls -lh best.onnx labels.txt
```

Kedua file harus ada. `labels.txt` berisi:
```
baju_lengan_panjang
baju_lengan_pendek
celana_panjang
celana_pendek
null
```

### 7. Jalankan server

```bash
# Cara 1: pakai script (otomatis load .env)
chmod +x run.sh
./run.sh

# Cara 2: manual dengan venv aktif
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Server siap ketika muncul:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Buka browser: **http://127.0.0.1:8000/ui**

---

## Kalibrasi Ukuran (Wajib sebelum estimasi ukuran bisa akurat)

`config.json` sudah ada di repo — kalau posisi kamera belum berubah, bisa langsung pakai.  
Kalau kamera diubah atau dipasang ulang, **kalibrasi ulang wajib dilakukan**.

### Opsi A — Via Web UI (Paling mudah)

1. Buka **http://127.0.0.1:8000/ui**
2. Klik tombol **"Kalibrasi"**
3. Upload foto papan pelipat kosong
4. Klik 4 sudut papan secara berurutan: **TL → TR → BR → BL**
5. Klik **"Simpan"**

### Opsi B — Via CLI (dari foto)

```bash
python calibration.py --image foto_pelipat.jpg
```

### Opsi C — Via CLI (kamera live, butuh display)

```bash
python calibration.py --perspective
```

---

## Environment Variables Lengkap

| Variable | Default | Keterangan |
|----------|---------|------------|
| `MODEL_PATH` | `./best.onnx` | Path ke model ONNX |
| `LABELS_PATH` | `./labels.txt` | Path ke file label |
| `CONF_THRESHOLD` | `0.75` | Threshold confidence YOLO (0.0–1.0) |
| `FOLD_ENABLED` | `true` | Aktifkan panduan lipatan |
| `SIZE_ESTIMATION_ENABLED` | `true` | Aktifkan estimasi ukuran |
| `SIZE_DEBUG_MODE` | `false` | Tampilkan nilai `lebar_cm` di overlay |
| `CONFIG_PATH` | `./config.json` | Path ke file kalibrasi |
| `MQTT_ENABLED` | `false` | Aktifkan publisher MQTT |
| `MQTT_HOST` | `localhost` | Alamat broker MQTT |
| `MQTT_PORT` | `1883` | Port broker MQTT |
| `MQTT_USER` | `clothbot` | Username MQTT |
| `MQTT_PASSWORD` | *(kosong)* | Password MQTT |
| `MQTT_TOPIC` | `clothbot/servo/command` | Topic MQTT untuk servo |
| `HOST` | `127.0.0.1` | Bind address server |
| `PORT` | `8000` | Port server |

---

## Endpoints API

| Method | Path | Keterangan |
|--------|------|------------|
| `GET` | `/ui` | Web UI |
| `GET` | `/health` | Status server |
| `GET` | `/docs` | Swagger docs otomatis |
| `POST` | `/predict` | Klasifikasi + fold + size estimation |
| `POST` | `/calibrate` | Kalibrasi skala dari foto |
| `POST` | `/calibrate-perspective` | Kalibrasi perspektif 4 titik |
| `GET` | `/metrics` | Statistik inferensi |

---

## Menjalankan di Mode Produksi

```bash
# Tanpa hot-reload, lebih stabil
./run.sh --no-reload

# Port kustom
PORT=9000 ./run.sh --no-reload

# Accessible dari luar (ganti host)
HOST=0.0.0.0 PORT=8000 ./run.sh --no-reload
```

---

## Troubleshooting

### `ModuleNotFoundError`
```bash
# Pastikan venv aktif
source .venv/bin/activate
pip install -r requirements.txt
```

### `config.json not found` / size estimation tidak aktif
```bash
# Lakukan kalibrasi dulu
python calibration.py --image foto_pelipat.jpg
```

### MQTT tidak terkoneksi
```bash
# Cek .env: MQTT_ENABLED=true dan MQTT_HOST sudah benar
# Test koneksi broker
python -c "import paho.mqtt.client as mqtt; c=mqtt.Client(); c.connect('localhost',1883,5); print('OK')"
```

### `best.onnx` tidak ada
File model tidak di-include di repo karena ukurannya besar (~145 MB).  
Salin dari Google Colab atau training output ke folder `gce/`.

---

## ESP32 Setup (Opsional — untuk aktuasi servo fisik)

1. Buka `clothbot_esp32/clothbot_esp32.ino` dengan Arduino IDE
2. Isi `secrets.h` dengan credentials:
   ```cpp
   #define WIFI_SSID_SECRET     "nama_wifi_kamu"
   #define WIFI_PASSWORD_SECRET "password_wifi"
   #define MQTT_HOST_SECRET     "ip_broker_mqtt"
   #define MQTT_PASS_SECRET     "password_mqtt"
   ```
3. Install library via Library Manager:
   - `PubSubClient` (Nick O'Leary)
   - `ArduinoJson` v7.x (Benoit Blanchon)
   - `Adafruit PWM Servo Driver Library`
4. Pilih board **ESP32 Dev Module**, upload
