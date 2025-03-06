#include <SoftwareSerial.h>


int i=0;
SoftwareSerial mySerial(10, 11); // RX, TX
String str;
void setup(){
 Serial.begin(115200);
 //Serial.print(char(169)); // Copyright Symbol
 //Serial.println(" Myengineeringstuffs.com");
 delay(2000);
 Serial.setTimeout(100);
 Serial.println(String(i));
 i++;
}

void loop()
{
  //Serial.print("H: ");

  //Serial.print("% ");
  //Serial.print(" T: ");

  //Serial.print(char(176));
  //Serial.println("C");
  //str =String('H')+String('T');
  //mySerial.println(str);
  while(!Serial.available()){}
  
  String str= Serial.readString();
  
  Serial.println(String(i));
  i++;

}
