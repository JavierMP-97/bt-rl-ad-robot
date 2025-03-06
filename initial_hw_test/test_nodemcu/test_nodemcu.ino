/*
 * HTTP Client GET Request
 * Copyright (c) 2018, circuits4you.com
 * All rights reserved.
 * https://circuits4you.com 
 * Connects to WiFi HotSpot. */
#include <ESP8266WiFi.h>
#include <WiFiClient.h> 
#include <ESP8266WebServer.h>
#include <ESP8266HTTPClient.h>

/* Set these to your desired credentials. */
const char* ssid = "";
const char* password = "";

//Web/Server address to read/write from 
const char* host = "";   //https://circuits4you.com website or IP address of server

HTTPClient http;    //Declare object of class HTTPClient
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
  delay(1000);
  WiFi.mode(WIFI_STA);        //This line hides the viewing of ESP as wifi hotspot
  
  WiFi.begin(ssid, password);     //Connect to your WiFi router
  //Serial.println("");

  //Serial.print("Connecting");
  // Wait for connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    //Serial.print(".");
  }

  String Link = "http://:5000/";
  
  http.begin(Link);     //Specify request destination
  http.setTimeout(100);
  //http.POST("2007");
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
  
  //GET Data
  //getData = "?status=" + "&station=";  //Note "?" added at front
  uint8_t i=0;
  uint8_t arduino[2];
  while(i<2){
    if(Serial.available()>0){
      arduino[i]=Serial.read();
      i++;
    }
  }
  
  //float arduino = Serial.parseFloat();

  //int httpCode = http.POST(String(arduino));            //Send the request
  http.POST(arduino,2);
  char payload = http.getString().charAt(0);    //Get the response payload
  
  //Serial.println(httpCode);   //Print HTTP return code
  
  //flushBuffer();
  
  Serial.print(payload);    //Print request response payload
}
//=======================================================================
