/*
 * test_ch1.ino — Test sweep servo CH1 via PCA9685
 * CH1 INVERT=true → kirim angle 0 = fisik 180°, kirim angle 180 = fisik 0°
 *
 * Sweep dilakukan PERLAHAN per 10° agar bisa lihat di titik mana servo berhenti/terhalang.
 *
 * Wiring: SDA=21, SCL=22, PCA9685 addr=0x40
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define SERVOMIN  102
#define SERVOMAX  512
#define SERVO_FREQ 50

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

uint16_t angleToPulse(int angle) {
  angle = constrain(angle, 0, 180);
  return (uint16_t)map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

// CH1 INVERT: effective = 180 - angle
void setCH1(int angle) {
  int effective = 180 - angle;
  Serial.printf("CH1 → angle=%d°  effective=%d°\n", angle, effective);
  pca.setPWM(1, 0, angleToPulse(effective));
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  pca.begin();
  pca.setOscillatorFrequency(27000000);
  pca.setPWMFreq(SERVO_FREQ);
  delay(10);

  Serial.println("=== Test CH1 ===");
  Serial.println("Ke home (0°)...");
  setCH1(0);
  delay(2000);
}

void loop() {
  // Sweep lambat 0° → 180° per 10°
  Serial.println("--- Sweep maju: 0° → 180° ---");
  for (int a = 0; a <= 180; a += 10) {
    setCH1(a);
    delay(300);
  }
  delay(1000);

  // Sweep balik 180° → 0° per 10°
  Serial.println("--- Sweep balik: 180° → 0° ---");
  for (int a = 180; a >= 0; a -= 10) {
    setCH1(a);
    delay(300);
  }
  delay(2000);
}
