#!/usr/bin/env python3
"""
test_ch3.py — Test gerak servo CH3 langsung via MQTT
Kirim perintah: CH3 dari home (0°) ke 180°, lalu balik home.

Jalankan:
    python3 test_ch3.py
"""

import json
import time
import paho.mqtt.client as mqtt

BROKER   = "45.58.168.24"
PORT     = 1883
USER     = "clothbot"
PASSWORD = "clothbot123"
TOPIC    = "clothbot/servo/command"

payload = {
    "label": "test_ch3",
    "steps": [
        # CH3 gerak ke 180°
        {"servos": [{"ch": 3, "angle": 180}], "delay_ms": 1000},
        # CH3 balik home
        {"servos": [{"ch": 3, "angle": 0}],   "delay_ms": 500},
    ]
}

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(USER, PASSWORD)
client.connect(BROKER, PORT, keepalive=10)
client.loop_start()
time.sleep(0.5)

info = client.publish(TOPIC, json.dumps(payload), qos=1)
info.wait_for_publish(timeout=5)
print(f"Published → {TOPIC}  rc={info.rc}  mid={info.mid}")

client.loop_stop()
client.disconnect()
