#include <SoftwareSerial.h>
#include <Wire.h>
//#include <NewPing.h>

//Line Tracking IO define
#include <Servo.h>  //servo library
Servo myservo;      // create servo object to control servo

#define ENA 10
#define ENB 5
#define IN1 9
#define IN2 8
#define IN3 7
#define IN4 6

uint8_t carSpeed = 175;

uint8_t speed_a = 175;

uint8_t speed_b = 175;


void forward(){
  speed_a=carSpeed;
  speed_b=carSpeed;
  analogWrite(ENA,speed_a); //enable L298n A channel
  analogWrite(ENB,speed_b); //enable L298n B channel
  digitalWrite(IN1,HIGH); //set IN1 hight level
  digitalWrite(IN2,LOW);  //set IN2 low level
  digitalWrite(IN3,HIGH);  //set IN3 low level
  digitalWrite(IN4,LOW); //set IN4 hight level
  //myservo.write(80);
  //Serial.println("Forward");//send message to serial monitor
}

void back(){
  speed_a=carSpeed;
  speed_b=carSpeed;
  analogWrite(ENA,speed_a);
  analogWrite(ENB,speed_b);
  digitalWrite(IN1,LOW);
  digitalWrite(IN2,HIGH);
  digitalWrite(IN3,LOW);
  digitalWrite(IN4,HIGH);
  //myservo.write(80);
  //Serial.println("Back");
}

void left(){
  speed_a=carSpeed;
  speed_b=carSpeed;
  analogWrite(ENA,speed_a);
  analogWrite(ENB,speed_b);
  digitalWrite(IN1,HIGH);
  digitalWrite(IN2,LOW);
  digitalWrite(IN3,LOW);
  digitalWrite(IN4,HIGH);
  //myservo.write(95);
  //Serial.println("Left");
}

void right(){
  speed_a=carSpeed;
  speed_b=carSpeed;
  analogWrite(ENA,speed_a);
  analogWrite(ENB,speed_b);
  digitalWrite(IN1,LOW);
  digitalWrite(IN2,HIGH);
  digitalWrite(IN3,HIGH);
  digitalWrite(IN4,LOW);
  //myservo.write(65);
  //Serial.println("Right");
}

void stop(){
   digitalWrite(ENA, LOW);
   digitalWrite(ENB, LOW);
   myservo.write(80);
   //Serial.println("Stop!");
} 

void accelerate(){
  digitalWrite(IN1,HIGH); //set IN1 hight level
  digitalWrite(IN2,LOW);  //set IN2 low level
  digitalWrite(IN3,HIGH);  //set IN3 low level
  digitalWrite(IN4,LOW);
  if(speed_a>=speed_b){
    speed_b=speed_a;
  }else{
    speed_a=speed_b;
  }
  speed_a+=10;
  speed_b+=10;

  if(speed_a>255) {speed_a=255;}
  if(speed_b>255) {speed_b=255;}
  
  analogWrite(ENB,speed_b);
  analogWrite(ENA,speed_a);
}

void brake(){
  if(speed_a>=speed_b){
    speed_b=speed_a;
  }else{
    speed_a=speed_b;
  }
  speed_a-=10;
  speed_b-=10;

  if(speed_a<0) {speed_a=0;}
  if(speed_b<0) {speed_b=0;}
  
  analogWrite(ENB,speed_b);
  analogWrite(ENA,speed_a);
}

void turn_right(float ratio){
  if(ratio >= 0){
    digitalWrite(IN1,HIGH); //set IN1 hight level
    digitalWrite(IN2,LOW);  //set IN2 low level
  }else{
    digitalWrite(IN1,LOW); //set IN1 hight level
    digitalWrite(IN2,HIGH);  //set IN2 low level
  }
  digitalWrite(IN3,HIGH);  //set IN3 low level
  digitalWrite(IN4,LOW); //set IN4 hight level
  
  if(speed_a>=speed_b){
    speed_b=speed_a;
  }else{
    speed_a=speed_b;
  }
  //speed_b=255;
  speed_a=speed_a*ratio;
  analogWrite(ENA,speed_a);
  analogWrite(ENB,speed_b);
  //myservo.write(95);
}

void turn_left(float ratio){
  digitalWrite(IN1,HIGH); //set IN1 hight level
  digitalWrite(IN2,LOW);  //set IN2 low level
  if(ratio >= 0){
    digitalWrite(IN3,HIGH);  //set IN3 low level
    digitalWrite(IN4,LOW); //set IN4 hight level    
  }else{
    digitalWrite(IN3,LOW);  //set IN3 low level
    digitalWrite(IN4,HIGH); //set IN4 hight level
  }

  if(speed_a>=speed_b){
    speed_b=speed_a;
  }else{
    speed_a=speed_b;
  }
  //speed_a=255;
  speed_b=speed_b*ratio;
  analogWrite(ENA,speed_a);
  analogWrite(ENB,speed_b);
  //myservo.write(65);
}

void keep_going(){}

void return_line(){
  
  stop();
  delay(2000);
  carSpeed = 175;
}

void actuate(char c){
  switch(c){
    case '0':
      turn_left(-0.1);
      break;
    case '1':
      turn_left(0.2);
      break;
    case '2':
      turn_left(0.5);
      break;
    case '3':
      forward();
      break;
    case '4':
      turn_right(0.5);
      break;
    case '5':
      turn_right(0.2);
      break;
    case '6':
      turn_right(-0.1);
      break;
    case '7':
      stop();
      break;
    case '8':
      return_line();
      break;
  }
}

void flushBuffer(){
  while(Serial.available()>0){
    Serial.read();
  }
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);//open serial and set the baudrate
  Wire.begin();
  pinMode(IN1,OUTPUT);//left//before useing io pin, pin mode must be set first 
  pinMode(IN2,OUTPUT);
  pinMode(IN3,OUTPUT);//right
  pinMode(IN4,OUTPUT);
  pinMode(ENA,OUTPUT);//left
  pinMode(ENB,OUTPUT);//right

  digitalWrite(IN1,HIGH); //set IN1 hight level
  digitalWrite(IN2,LOW);  //set IN2 low level
  digitalWrite(IN3,LOW);  //set IN3 low level
  digitalWrite(IN4,HIGH); //set IN4 hight level

  Serial.begin(115200);
  //pinMode(10,INPUT);
  //pinMode(4,INPUT);
  //pinMode(2,INPUT);
  //myservo.attach(3);
  //myservo.write(80);
  delay(1000);
  //myservo.detach();
  delay(2000);
  
}

void loop() {
  // put your main code here, to run repeatedly:
  char buf[10];
  
  uint8_t count = 0;
  Wire.requestFrom(0x11, 10);
  while (Wire.available() ) { // slave may send less than requested
    char c = Wire.read(); // receive a byte as character
    buf[count] = c;
    count += 1;
    
  }
  for(int i=0; i<10; i++){
    Serial.write(buf[i]);
  }
  
  int counter = 0;
  while(Serial.available()<1){
    if(counter < 100){
      counter++;
      delay(1); 
    }else if(counter == 100){
      actuate('7');
      counter++;
    }
  }
   
  char nodemcu = Serial.read();
  
  actuate(nodemcu);
}
