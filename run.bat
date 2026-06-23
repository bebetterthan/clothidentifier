@echo off
REM ============================================================
REM  run.bat — Start ClothBot API server (Windows)
REM
REM  Usage:
REM    run.bat               -> hot-reload di 127.0.0.1:8000
REM    run.bat --no-reload   -> tanpa hot-reload (production-like)
REM    run.bat --port 9000   -> custom port
REM    run.bat --host 0.0.0.0 --port 8000  -> expose ke LAN
REM ============================================================
setlocal enabledelayedexpansion

REM ── Pindah ke direktori script ────────────────────────────
cd /d "%~dp0"

REM ── Default konfigurasi ───────────────────────────────────
set HOST=127.0.0.1
set PORT=8000
set RELOAD_FLAG=--reload

set MQTT_ENABLED=false
set FOLD_ENABLED=true
set SIZE_ESTIMATION_ENABLED=true
set SIZE_DEBUG_MODE=false

REM ── Load .env jika ada ────────────────────────────────────
if exist ".env" (
    echo [run.bat] Loading .env ...
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set line=%%A
        REM Skip baris komentar (#) dan baris kosong
        if not "!line:~0,1!"=="#" (
            if not "%%A"=="" (
                set "%%A=%%B"
            )
        )
    )
    echo [run.bat] .env loaded
) else (
    echo [run.bat] No .env found — using defaults
    echo [run.bat]   MQTT disabled, Fold enabled, Size enabled
)

REM ── Parse argumen command line ────────────────────────────
:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="--no-reload" (
    set RELOAD_FLAG=
    shift
    goto parse_args
)
if /i "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--host" (
    set HOST=%~2
    shift
    shift
    goto parse_args
)
shift
goto parse_args
:end_parse

REM ── Cari uvicorn: venv dulu, lalu PATH ───────────────────
set UVICORN=

if exist ".venv\Scripts\uvicorn.exe" (
    set UVICORN=.venv\Scripts\uvicorn.exe
    echo [run.bat] Using venv uvicorn
    goto found_uvicorn
)

where uvicorn >nul 2>&1
if %errorlevel%==0 (
    set UVICORN=uvicorn
    echo [run.bat] Using system uvicorn
    goto found_uvicorn
)

echo.
echo [run.bat] ERROR: uvicorn tidak ditemukan.
echo           Jalankan salah satu perintah berikut:
echo.
echo           Jika pakai venv:
echo             python -m venv .venv
echo             .venv\Scripts\pip install -r requirements.txt
echo.
echo           Jika install global:
echo             pip install -r requirements.txt
echo.
pause
exit /b 1

:found_uvicorn

REM ── Cek config.json ───────────────────────────────────────
if not exist "config.json" (
    if /i "%SIZE_ESTIMATION_ENABLED%"=="true" (
        echo.
        echo [run.bat] WARN: config.json tidak ditemukan.
        echo           Size estimation akan nonaktif sampai kalibrasi selesai.
        echo           Opsi kalibrasi:
        echo             1. Dari foto  : python calibration.py --image foto_pelipat.jpg
        echo             2. Kamera live: python calibration.py --camera
        echo             3. Via API    : POST http://%HOST%:%PORT%/calibrate
        echo.
    )
)

REM ── Info startup ──────────────────────────────────────────
echo.
echo [run.bat] ==========================================
echo [run.bat]  ClothBot API - Windows
echo [run.bat] ==========================================
echo [run.bat]  App  : http://%HOST%:%PORT%
echo [run.bat]  UI   : http://%HOST%:%PORT%/ui
echo [run.bat]  Docs : http://%HOST%:%PORT%/docs
echo [run.bat]  MQTT : %MQTT_ENABLED%
echo [run.bat]  Fold : %FOLD_ENABLED%
echo [run.bat]  Size : %SIZE_ESTIMATION_ENABLED%
if defined RELOAD_FLAG (
    echo [run.bat]  Mode : development ^(hot-reload aktif^)
) else (
    echo [run.bat]  Mode : production ^(no-reload^)
)
echo [run.bat] ==========================================
echo.

REM ── Jalankan server ───────────────────────────────────────
%UVICORN% app:app ^
    --host %HOST% ^
    --port %PORT% ^
    --workers 1 ^
    %RELOAD_FLAG%

REM ── Tangkap error jika server crash ───────────────────────
if %errorlevel% neq 0 (
    echo.
    echo [run.bat] Server berhenti dengan error code %errorlevel%
    echo           Periksa log di atas untuk detail.
    pause
)

endlocal
