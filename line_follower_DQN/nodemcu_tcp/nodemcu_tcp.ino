/*
 * HTTP Client GET Request
 * Copyright (c) 2018, circuits4you.com
 * All rights reserved.
 * https://circuits4you.com 
 * Connects to WiFi HotSpot. */
#include <ESP8266WiFi.h>
#include <ESP8266WiFiMulti.h>


/* Set these to your desired credentials. */
const char* ssid_pc = "";
const char* password_pc = "";
const char* ssid_router = "";
const char* password_router = "";
const char* ssid_router_2 = "";
const char* password_router_2 = "";

//Web/Server address to read/write from
const char* host_pc = ""; 
const char* host_router = "";
const uint16_t port = 5000;

WiFiClient client;    //Declare object of class HTTPClient
//=======================================================================
//                    Power on setup
//=======================================================================

void flushBuffer(){
  while(Serial.available()>0){
    Serial.read();
  }
}

void setup() {
  delay(1000);
  Serial.begin(115200);
  WiFi.mode(WIFI_OFF);        //Prevents reconnection issue (taking too long to connect)
  delay(100);
  Serial.setTimeout(30000);
  WiFi.mode(WIFI_STA);        //This line hides the viewing of ESP as wifi hotspot
  delay(100);
  WiFi.begin(ssid_pc, password_pc);     //Connect to your WiFi router
  //Serial.println("");

  //Serial.print("Connecting");
  // Wait for connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(100);
    //Serial.print(".");
  }
  
  client.connect(host_pc, port);     //Specify request destination

  //If connection successful show IP address in serial monitor
  /*Serial.println("");
  Serial.print("Connected to ");
  Serial.println(ssid);
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());  //IP address assigned to your ESP
  */
}

//=======================================================================
//                    Main Program Loop
//=======================================================================
void loop() {
  uint8_t i=0;
  char arduino[10];
  while(i<10){
    if(Serial.available()>0){
      arduino[i]=Serial.read();
      i++;
    }
  }

  client.write(arduino,10);
  
  while(client.available()==0);
  
  //flushBuffer();

  char pc = client.read();
  
  Serial.write(pc);    //Print request response
}
//=======================================================================
