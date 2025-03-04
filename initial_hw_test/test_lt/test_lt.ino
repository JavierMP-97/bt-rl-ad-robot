#include <Wire.h>
int count = 0;
int raw_result[10];
int analog_result[5];
int high_byte;
int low_byte;
void setup() {
  Wire.begin();        // join i2c bus (address optional for master)
  Serial.begin(9600);  // start serial for output
  Serial.println("Ready");
  delay(1000);
}
void loop() {
  Wire.requestFrom(0x11, 10);    // request 6 bytes from slave device #8
  while (Wire.available()) { // slave may send less than requested
    int c = Wire.read(); // receive a byte as character
    Serial.print(c, HEX);        // print the character
    Serial.print("  ");
    raw_result[count] = c;
    count += 1;
  }
  Serial.println(' ');
  count = 0;
  for (int i = 0; i < 5; i++) {
    high_byte = raw_result[i * 2] << 8;
    low_byte = raw_result[i * 2 + 1];
    analog_result[i] = high_byte + low_byte;
    Serial.print(analog_result[i]);
    Serial.print("  ");
  }
  Serial.println(' ');
  delay(500);
}
