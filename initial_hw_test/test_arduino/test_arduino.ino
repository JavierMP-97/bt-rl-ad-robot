#include <SoftwareSerial.h>

#include <NewPing.h>

//Line Tracking IO define
#include <Servo.h>  //servo library
Servo myservo;      // create servo object to control servo

#define LT_R !digitalRead(10)
#define LT_M !digitalRead(4)
#define LT_L !digitalRead(2)

#define ENA 5
#define ENB 6
#define IN1 7
#define IN2 8
#define IN3 9
#define IN4 11

uint8_t carSpeed = 200;

uint8_t speed_a = 0;

uint8_t speed_b = 0;

uint8_t Echo = A4;  
uint8_t Trig = A5;

uint8_t last_line=1; 

NewPing sonar(Trig, Echo, 200);
uint8_t md = 0;

void forward(){
  speed_a=carSpeed;
  speed_b=carSpeed;
  analogWrite(ENA,speed_a); //enable L298n A channel
  analogWrite(ENB,speed_b); //enable L298n B channel
  digitalWrite(IN1,HIGH); //set IN1 hight level
  digitalWrite(IN2,LOW);  //set IN2 low level
  digitalWrite(IN3,LOW);  //set IN3 low level
  digitalWrite(IN4,HIGH); //set IN4 hight level
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
  digitalWrite(IN3,HIGH);
  digitalWrite(IN4,LOW);
  //myservo.write(80);
  //Serial.println("Back");
}

void left(){
  speed_a=carSpeed;
  speed_b=carSpeed;
  analogWrite(ENA,speed_a);
  analogWrite(ENB,speed_b);
  digitalWrite(IN1,LOW);
  digitalWrite(IN2,HIGH);
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
  digitalWrite(IN1,HIGH);
  digitalWrite(IN2,LOW);
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
  digitalWrite(IN3,LOW);  //set IN3 low level
  digitalWrite(IN4,HIGH);
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

void turn_left(){
  if(speed_a>=speed_b){
    speed_b=speed_a;
  }else{
    speed_a=speed_b;
  }
  speed_b=255;
  speed_a=speed_a*0.3;
  analogWrite(ENA,speed_a);
  analogWrite(ENB,speed_b);
  //myservo.write(95);
}

void turn_right(){
  if(speed_a>=speed_b){
    speed_b=speed_a;
  }else{
    speed_a=speed_b;
  }
  speed_a=255;
  speed_b=speed_b*0.3;
  analogWrite(ENA,speed_a);
  analogWrite(ENB,speed_b);
  //myservo.write(65);
}

void keep_going(){}

void return_line(){
  
  if (last_line == 0){
    left();
  }else if (last_line == 2){
    right();
  }else if (last_line == 1){
    carSpeed = 150;
    back();
  }
  while (LT_M == 0/* & LT_R == 0 & LT_L == 0*/){}
  stop();
  carSpeed = 200;
}

void actuate(char c){
  switch(c){
    case '0':
      left();
      break;
    case '1':
      turn_left();
      break;
    case '2':
      forward();
      break;
    case '3':
      turn_right();
      break;
    case '4':
      right();
      break;
    case '5':
      stop();
      break;
    case '6':
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
  pinMode(10,INPUT);
  pinMode(4,INPUT);
  pinMode(2,INPUT);
  myservo.attach(3);
  pinMode(Echo, INPUT);    
  pinMode(Trig, OUTPUT);
  myservo.write(80);
  delay(1000);
  myservo.detach();
  delay(2000);
  
}

void loop() {
  // put your main code here, to run repeatedly:
  md = sonar.ping_cm();
  uint8_t line_l=LT_L;
  uint8_t line_m=LT_M;
  uint8_t line_r=LT_R;
  if(md==0)
    md=200;
  if (line_l == 1){
    last_line = 0;
  }else if (line_r == 1){
    last_line = 2;
  }else if (line_m == 1){
    last_line = 1;
  }
  char buf[2];
  buf[0]=md;
  buf[1]= LT_L*4+LT_M*2+LT_R+1;
  //flushBuffer();
  //Serial.write(buf);
  Serial.write(buf[0]);
  Serial.write(buf[1]);
  //delay(100);
  
  while(Serial.available()<1); 
  char nodemcu = Serial.read();
  
  actuate(nodemcu);
}
