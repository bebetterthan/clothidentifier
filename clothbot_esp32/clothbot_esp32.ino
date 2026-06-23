/*
 * ClothBot ESP32 — MQTT Subscriber + PCA9685 Servo Controller
 * ============================================================
 * v2.0 — non-blocking state machine, watchdog, MAC-based client ID
 *
 * Topic  : clothbot/servo/command  (QoS 1)
 * Payload: {"label":"...","steps":[{"servos":[{"ch":N,"angle":N}],"delay_ms":N},...]}
 *
 * Hardware
 * --------
 *   ESP32  ↔  PCA9685  (I2C: SDA=21 SCL=22, addr=0x40)
 *   PCA9685 CH0 → Right side arm   (home   0°)
 *   PCA9685 CH1 → Left side arm    (home 180°)
 *   PCA9685 CH2 → Bottom fold R    (home   0°)
 *   PCA9685 CH3 → Bottom fold L    (home   0°)
 *
 * Library dependencies (install via Library Manager)
 * ---------------------------------------------------
 *   PubSubClient          — Nick O'Leary
 *   ArduinoJson           — Benoit Blanchon  (v7.x)
 *   Adafruit PWM Servo Driver Library
 *   Adafruit BusIO        (required by above)
 *
 * Credentials → edit WIFI_SSID / WIFI_PASSWORD di bawah
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <esp_task_wdt.h>

// ── WiFi credentials ───────────────────────────────────────────────────────
const char* WIFI_SSID     = "DORA";
const char* WIFI_PASSWORD = "peler321";

// ── MQTT broker ────────────────────────────────────────────────────────────
const char* MQTT_HOST  = "45.58.168.24";
const int   MQTT_PORT  = 1883;
const char* MQTT_USER  = "clothbot";
const char* MQTT_PASS  = "clothbot123";  // password broker
const char* MQTT_TOPIC = "clothbot/servo/command";

// ── PCA9685 servo calibration ─────────────────────────────────────────────
// 50 Hz → 20 ms period → 4096 ticks
// 500 us → 500*4096/20000  = 102 ticks  (0°)
// 2500 us → 2500*4096/20000 = 512 ticks (180°)
#define SERVOMIN    102    // ~500 us — safe minimum for most servos (SG90/MG996R)
#define SERVOMAX    512    // ~2500 us — safe maximum, fine-tune if needed
#define SERVO_FREQ   50    // Hz

// ── Watchdog ───────────────────────────────────────────────────────────────
#define WDT_TIMEOUT_S  30

// ── Reconnect intervals ────────────────────────────────────────────────────
const unsigned long WIFI_RECONNECT_MS = 10000;
const unsigned long MQTT_RECONNECT_MS =  5000;

// ── Home positions ─────────────────────────────────────────────────────────
struct HomePos { uint8_t ch; int angle; };
static const HomePos HOME[]  = { {0, 0}, {1, 0}, {2, 0}, {3, 0} };
static const int     HOME_N  = sizeof(HOME) / sizeof(HOME[0]);

// ── Per-channel direction invert ───────────────────────────────────────────
// true  = servo dipasang terbalik → angle di-flip: effectiveAngle = 180 - angle
// false = arah normal
static const bool INVERT[16] = {
  false,  // CH0 — arah normal (dulu CH1)
  true,   // CH1 — kebalik (dulu CH0)
  false,  // CH2
  false,  // CH3
};

// ── Servo sequence state machine ───────────────────────────────────────────
static const int MAX_STEPS  = 10;
static const int MAX_SERVOS =  4;

struct ServoCmd { uint8_t ch; int angle; };
struct Step     { ServoCmd servos[MAX_SERVOS]; uint8_t numServos; uint32_t delayMs; };

static Step          seqSteps[MAX_STEPS];
static int           seqLen      = 0;
static int           seqCurrent  = 0;
static bool          seqRunning  = false;
static bool          stepMoved   = false;
static unsigned long stepStartMs = 0;

// ── Globals ────────────────────────────────────────────────────────────────
Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

WiFiClient   netClient;
PubSubClient mqttClient(netClient);

char         mqttClientId[32];
unsigned long lastWifiAttempt = 0;
unsigned long lastMqttAttempt = 0;

// ── Servo helpers ──────────────────────────────────────────────────────────

uint16_t angleToPulse(int angle) {
  angle = constrain(angle, 0, 180);
  return (uint16_t)map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

void setServo(uint8_t ch, int angle) {
  if (ch > 15) {
    Serial.printf("  [WARN] CH%u > 15, skipped.\n", ch);
    return;
  }
  int effective = (ch < 16 && INVERT[ch]) ? (180 - angle) : angle;
  pca.setPWM(ch, 0, angleToPulse(effective));
}

void goHome() {
  Serial.println("[SERVO] Homing all channels.");
  for (int i = 0; i < HOME_N; i++) setServo(HOME[i].ch, HOME[i].angle);
  delay(500);  // blocking OK here — still inside setup()
}

// ── Non-blocking servo state machine ──────────────────────────────────────

void runSequence() {
  if (!seqRunning) return;

  if (seqCurrent >= seqLen) {
    seqRunning = false;
    Serial.println("[SERVO] Sequence complete.");
    return;
  }

  Step& s = seqSteps[seqCurrent];

  if (!stepMoved) {
    // Kick all servos for this step at once, then record start time
    Serial.printf("  Step %d/%d  (delay %u ms)\n", seqCurrent + 1, seqLen, s.delayMs);
    for (int i = 0; i < s.numServos; i++) {
      setServo(s.servos[i].ch, s.servos[i].angle);
      Serial.printf("    CH%u → %d°\n", s.servos[i].ch, s.servos[i].angle);
    }
    stepStartMs = millis();
    stepMoved   = true;
  }

  // Advance to next step after delay — no blocking
  if (millis() - stepStartMs >= s.delayMs) {
    seqCurrent++;
    stepMoved = false;
  }
}

void i2cScan() {
  Serial.println("[I2C] Scanning...");
  int found = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.printf("[I2C] Device found at 0x%02X", addr);
      if (addr == 0x40) Serial.print("  ← PCA9685");
      Serial.println();
      found++;
    }
  }
  if (found == 0) Serial.println("[I2C] No devices found! Check SDA/SCL wiring.");
}

/** Sweep CH0 from 0→90→0 to verify PCA9685 + servo power */
void servoSelfTest() {
  Serial.println("[TEST] Servo self-test: CH0 sweep 0→90→0");
  setServo(0, 0);   delay(400);
  setServo(0, 90);  delay(600);
  setServo(0, 0);   delay(400);
  Serial.println("[TEST] Done. If CH0 did NOT move:");
  Serial.println("[TEST]   1. Check V+ pin on PCA9685 (servo power, 5-6V)");
  Serial.println("[TEST]   2. Check SERVOMIN/SERVOMAX values for your servo model");
  Serial.println("[TEST]   3. Verify I2C address above matches 0x40");
}



void onMessage(char* topic, byte* payload, unsigned int length) {
  Serial.printf("\n[MQTT] ← %s  (%u bytes)\n", topic, length);

  if (seqRunning) {
    Serial.println("[MQTT] Sequence in progress — new command ignored.");
    return;
  }

  JsonDocument doc;
  DeserializationError err = deserializeJson(doc, payload, length);
  if (err) {
    Serial.printf("[MQTT] JSON parse error: %s\n", err.c_str());
    return;
  }
  if (doc.overflowed()) {
    Serial.println("[MQTT] JSON overflow — payload too large, increase setBufferSize.");
    return;
  }

  const char* label = doc["label"] | "unknown";
  JsonArray   steps = doc["steps"];

  if (steps.isNull() || steps.size() == 0) {
    Serial.println("[MQTT] No 'steps' in payload.");
    return;
  }

  // Copy step data into static buffer before callback returns
  // (payload pointer is only valid for the duration of this function)
  seqLen = 0;
  for (JsonObject step : steps) {
    if (seqLen >= MAX_STEPS) break;
    Step& s   = seqSteps[seqLen];
    s.delayMs   = step["delay_ms"] | 500;
    s.numServos = 0;
    for (JsonObject sv : step["servos"].as<JsonArray>()) {
      if (s.numServos >= MAX_SERVOS) break;
      uint8_t ch = sv["ch"] | 0;
      if (ch > 15) {
        Serial.printf("    [WARN] CH%u > 15 in payload, skipped.\n", ch);
        continue;
      }
      s.servos[s.numServos++] = { ch, sv["angle"] | 0 };
    }
    seqLen++;
  }

  Serial.printf("[SERVO] Label: %s  Steps loaded: %d\n", label, seqLen);
  seqCurrent = 0;
  stepMoved  = false;
  seqRunning = true;  // hand off execution to loop() / runSequence()
}

// ── WiFi (non-blocking) ────────────────────────────────────────────────────

void checkWiFi() {
  static bool prevConnected = false;
  bool connected = (WiFi.status() == WL_CONNECTED);

  if (connected && !prevConnected) {
    prevConnected = true;
    Serial.printf("[WiFi] Connected  IP: %s\n", WiFi.localIP().toString().c_str());
  } else if (!connected && prevConnected) {
    prevConnected = false;
    Serial.println("[WiFi] Connection lost.");
  }

  if (connected) return;

  unsigned long now = millis();
  if (now - lastWifiAttempt < WIFI_RECONNECT_MS) return;
  lastWifiAttempt = now;

  Serial.printf("[WiFi] Reconnecting to %s…\n", WIFI_SSID);
  WiFi.disconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

// ── MQTT reconnect (non-blocking) ──────────────────────────────────────────

void checkMQTT() {
  if (mqttClient.connected()) return;
  if (WiFi.status() != WL_CONNECTED) return;

  unsigned long now = millis();
  if (now - lastMqttAttempt < MQTT_RECONNECT_MS) return;
  lastMqttAttempt = now;

  Serial.printf("[MQTT] Connecting to %s:%d  id=%s\n", MQTT_HOST, MQTT_PORT, mqttClientId);
  if (mqttClient.connect(mqttClientId, MQTT_USER, MQTT_PASS)) {
    mqttClient.subscribe(MQTT_TOPIC, 1);  // QoS 1
    Serial.printf("[MQTT] Subscribed → %s\n", MQTT_TOPIC);
  } else {
    Serial.printf("[MQTT] Failed  rc=%d\n", mqttClient.state());
  }
}

// ── Setup ──────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("\n=== ClothBot ESP32 v2.0 ===");

  // Hardware watchdog — resets device if loop() hangs
  // ESP-IDF 5.x (Arduino core 3.x): use esp_task_wdt_reconfigure()
  // ESP-IDF 4.x (Arduino core 2.x): use esp_task_wdt_init()
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
  esp_task_wdt_config_t wdt_cfg = {
    .timeout_ms     = WDT_TIMEOUT_S * 1000,
    .idle_core_mask = 0,
    .trigger_panic  = true,
  };
  esp_task_wdt_reconfigure(&wdt_cfg);
#else
  esp_task_wdt_init(WDT_TIMEOUT_S, true);
#endif
  esp_task_wdt_add(NULL);
  Serial.printf("[WDT] Watchdog armed (%d s).\n", WDT_TIMEOUT_S);

  // PCA9685
  Wire.begin();  // SDA=21, SCL=22 (ESP32 default)
  i2cScan();     // print all I2C devices — confirms PCA9685 address
  pca.begin();
  pca.setOscillatorFrequency(27000000);  // tune if servos jitter
  pca.setPWMFreq(SERVO_FREQ);
  delay(10);
  Serial.println("[PCA9685] Initialized at 0x40.");

  // Move all servos to known home positions before taking commands
  goHome();

  // Quick sweep to confirm servo + PCA9685 are physically working
  servoSelfTest();

  // Build unique client ID from hardware MAC address
  snprintf(mqttClientId, sizeof(mqttClientId), "clothbot-%llx", ESP.getEfuseMac());
  Serial.printf("[MQTT] Client ID: %s\n", mqttClientId);

  // Start WiFi (non-blocking — loop() handles waiting + retries)
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("[WiFi] Connecting to %s…\n", WIFI_SSID);
  lastWifiAttempt = millis();

  // MQTT client config
  mqttClient.setServer(MQTT_HOST, MQTT_PORT);
  mqttClient.setCallback(onMessage);
  mqttClient.setBufferSize(2048);
  mqttClient.setKeepAlive(60);    // 60s keepalive — reduces disconnect behind NAT
  mqttClient.setSocketTimeout(15); // 15s TCP timeout

  Serial.println("[Setup] Done. Waiting for WiFi…");
}

// ── Loop ───────────────────────────────────────────────────────────────────

void loop() {
  esp_task_wdt_reset();  // pet the watchdog

  checkWiFi();    // reconnect WiFi non-blocking
  checkMQTT();    // reconnect MQTT non-blocking
  mqttClient.loop();
  runSequence();  // advance servo state machine non-blocking
}
