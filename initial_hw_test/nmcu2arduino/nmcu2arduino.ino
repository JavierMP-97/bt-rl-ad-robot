void setup() {
  // Open serial communications and wait for port to open:
  Serial.begin(115200);
  //while (!Serial) {
    //; // wait for serial port to connect. Needed for native USB port only
  //}
  Serial.setTimeout(100);
}

void loop() { // run over and over
  while(!Serial.available()){} 
  Serial.println("NodeMCU"+String(Serial.parseInt()));
}
