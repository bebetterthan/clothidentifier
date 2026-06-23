/*
 * test_ch3.ino — Test gerak servo CH3 via PCA9685
 * CH3 sweep: home → 180° → home, terus-menerus
 *
 * Wiring: SDA=21, SCL=22, PCA9685 addr=0x40
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define SERVOMIN  102   // ~500 us (0°)
#define SERVOMAX  512   // ~2500 us (180°)
#define SERVO_FREQ 50

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

uint16_t angleToPulse(int angle) {
  angle = constrain(angle, 0, 180);
  return (uint16_t)map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  pca.begin();
  pca.setOscillatorFrequency(27000000);
  pca.setPWMFreq(SERVO_FREQ);
  delay(10);
  Serial.println("Test CH3 dimulai...");
}

void loop() {
  Serial.println("CH3 → 180°");
  pca.setPWM(3, 0, angleToPulse(180));
  delay(1500);

  Serial.println("CH3 → 0° (home)");
  pca.setPWM(3, 0, angleToPulse(0));
  delay(1500);
}
