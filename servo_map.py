"""
servo_map.py — Clothing label → PCA9685 servo sequence mapping
ClothBot Folding Machine | 4-channel PCA9685 via I2C

Channel layout
--------------
CH0  Right side arm   (fold right → left)   home=0°,   fold varies
CH1  Left side arm    (fold left → right)    home=180°, fold varies
CH2  Bottom fold R    (fold up)              home=0°,   fold varies
CH3  Bottom fold L    (fold up)              home=0°,   fold varies

Fold sequence rationale
-----------------------
Baju (shirt):   side arms first → body/bottom fold
                Avoids fabric collision at waist area.

Celana (pants): leg fold first → side fold
                Legs must be aligned before side compression.

Each label maps to a list of "steps". The ESP32 should:
  1. Iterate steps in order.
  2. Move all servos in a step simultaneously to their target angles.
  3. Wait `delay_ms` before starting the next step.

NOTE: ESP32-side subscriber uses Arduino PubSubClient + ArduinoJson.
      Payload structure: {"label": "...", "steps": [{...}, ...]}
      ESP32 iterates msg["steps"], drives PCA9685 channels, then delays.
"""

from typing import TypedDict


class ServoAngle(TypedDict):
    """A single servo channel and its target angle."""
    ch: int       # PCA9685 channel number (0-15)
    angle: int    # Target angle in degrees (0-180)


class ServoStep(TypedDict):
    """One timed step: move all listed servos, then wait delay_ms."""
    servos: list[ServoAngle]
    delay_ms: int


class LabelMapping(TypedDict):
    """Full servo sequence for one clothing label."""
    steps: list[ServoStep]


# ---------------------------------------------------------------------------
# Home positions — all channels return here in the final step
# ---------------------------------------------------------------------------
_HOME: list[ServoAngle] = [
    {"ch": 0, "angle": 0},
    {"ch": 1, "angle": 0},
    {"ch": 2, "angle": 0},
    {"ch": 3, "angle": 0},
]

# ---------------------------------------------------------------------------
# SERVO_MAP
# Keys must exactly match the class names in labels.txt.
# "null" is intentionally absent — it must never trigger a publish.
# ---------------------------------------------------------------------------
SERVO_MAP: dict[str, LabelMapping] = {

    "baju_lengan_panjang": {
        "steps": [
            # Step 1 — fold lengan kanan (CH1)
            {"servos": [{"ch": 1, "angle": 180}], "delay_ms": 600},
            # Step 2 — CH1 balik home, jeda 5 detik (beri waktu arus stabil)
            {"servos": [{"ch": 1, "angle": 0}], "delay_ms": 5000},
            # Step 3 — fold lengan kiri (CH0)
            {"servos": [{"ch": 0, "angle": 180}], "delay_ms": 600},
            # Step 4 — CH0 balik home
            {"servos": [{"ch": 0, "angle": 0}], "delay_ms": 600},
            # Step 5 — fold badan bawah
            {"servos": [{"ch": 2, "angle": 180}, {"ch": 3, "angle": 180}], "delay_ms": 600},
            # Step 6 — semua ke home
            {"servos": _HOME, "delay_ms": 400},
        ]
    },

    "baju_lengan_pendek": {
        "steps": [
            # Step 1 — fold lengan kanan (CH1)
            {"servos": [{"ch": 1, "angle": 180}], "delay_ms": 600},
            # Step 2 — CH1 balik home, jeda 5 detik (beri waktu arus stabil)
            {"servos": [{"ch": 1, "angle": 0}], "delay_ms": 5000},
            # Step 3 — fold lengan kiri (CH0)
            {"servos": [{"ch": 0, "angle": 180}], "delay_ms": 600},
            # Step 4 — CH0 balik home
            {"servos": [{"ch": 0, "angle": 0}], "delay_ms": 600},
            # Step 5 — fold badan bawah
            {"servos": [{"ch": 2, "angle": 180}, {"ch": 3, "angle": 180}], "delay_ms": 600},
            # Step 6 — semua ke home
            {"servos": _HOME, "delay_ms": 400},
        ]
    },

    "celana_panjang": {
        "steps": [
            # Step 1 — fold sisi kanan (CH1)
            {"servos": [{"ch": 1, "angle": 180}], "delay_ms": 600},
            # Step 2 — CH1 balik home, jeda 5 detik
            {"servos": [{"ch": 1, "angle": 0}], "delay_ms": 5000},
            # Step 3 — fold sisi kiri (CH0)
            {"servos": [{"ch": 0, "angle": 180}], "delay_ms": 600},
            # Step 4 — CH0 balik home
            {"servos": [{"ch": 0, "angle": 0}], "delay_ms": 600},
            # Step 5 — fold kaki bawah (CH3) terlebih dahulu
            {"servos": [{"ch": 3, "angle": 180}], "delay_ms": 2000},
            # Step 6 — fold kaki atas (CH2) setelah jeda 2 detik
            {"servos": [{"ch": 2, "angle": 180}], "delay_ms": 700},
            # Step 7 — semua ke home
            {"servos": _HOME, "delay_ms": 400},
        ]
    },

    "celana_pendek": {
        "steps": [
            # Step 1 — fold sisi kanan (CH1)
            {"servos": [{"ch": 1, "angle": 180}], "delay_ms": 600},
            # Step 2 — CH1 balik home, jeda 5 detik
            {"servos": [{"ch": 1, "angle": 0}], "delay_ms": 5000},
            # Step 3 — fold sisi kiri (CH0)
            {"servos": [{"ch": 0, "angle": 180}], "delay_ms": 600},
            # Step 4 — CH0 balik home
            {"servos": [{"ch": 0, "angle": 0}], "delay_ms": 600},
            # Step 5 — fold kaki bawah (CH3) terlebih dahulu
            {"servos": [{"ch": 3, "angle": 180}], "delay_ms": 2000},
            # Step 6 — fold kaki atas (CH2) setelah jeda 2 detik
            {"servos": [{"ch": 2, "angle": 180}], "delay_ms": 700},
            # Step 7 — semua ke home
            {"servos": _HOME, "delay_ms": 400},
        ]
    },

}

# Convenience: total step count per label for validation / logging
STEP_COUNTS: dict[str, int] = {
    label: len(mapping["steps"]) for label, mapping in SERVO_MAP.items()
}
