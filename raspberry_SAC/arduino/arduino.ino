//#include <SoftwareSerial.h>
#include <Wire.h>
#include <math.h>
//#include <NewPing.h>

//Line Tracking IO define
//#include <Servo.h>  //servo library
//Servo myservo;      // create servo object to control servo

#define ENA 10
#define ENB 5
#define IN1 9
#define IN2 8
#define IN3 7
#define IN4 6

float carSpeed = 0;

uint8_t speed_a = 0;

uint8_t speed_b = 0;

uint8_t minCarSpeed = 175;

uint8_t maxCarSpeed = 255;

uint8_t minSteerSpeed = 125;


float sigmoide(float x)
{
  return (1/(1+exp(-x-5)));
}

float logit(float x)
{
  return -log((1/x)-1)+5;
}

void stop(){
   digitalWrite(ENA, LOW);
   digitalWrite(ENB, LOW);
   //myservo.write(80);
   //Serial.println("Stop!");
} 

void safeStop(){
  carSpeed = 0;
  while((speed_a > 0) || (speed_b > 0)){
    if(speed_a > 0)
      speed_a = speed_a - 1;
    if(speed_b > 0)
      speed_b = speed_b - 1;
    analogWrite(ENA,speed_a);
    analogWrite(ENB,speed_b);
    delay(2);
  }
}

void actuate2(float throttle, float steering){
  if (throttle < 0){
    carSpeed = carSpeed + throttle/30;
  }else if (throttle > carSpeed) {
    float unprocessedSpeed = logit(carSpeed*10);
    carSpeed = sigmoide((throttle/10)+unprocessedSpeed)/10;
  } else {
    float normDiff = (throttle - carSpeed) / carSpeed;
    carSpeed = carSpeed + normDiff/200;
  }


  float wheelDiff;
  if (carSpeed > 0.15){
    wheelDiff = (1 - carSpeed)/0.85;
  } else { 
    wheelDiff = 1;
  }
  float wheelDiffSteer = (0.05 + 0.1*wheelDiff) * steering;

  float rightWheel = min(0.999,max(0, carSpeed + wheelDiffSteer));
  float leftWheel = min(0.999,max(0, carSpeed - wheelDiffSteer));

  //speed_a = (uint8_t)(leftWheel * 256);
  //speed_b = (uint8_t)(rightWheel * 256);
  speed_a = (uint8_t)((leftWheel * 176) + 80);
  speed_b = (uint8_t)((rightWheel * 176) + 80);

  //Serial.write(speed_a);
  //Serial.write(speed_b);
  
  analogWrite(ENA,speed_a);
  analogWrite(ENB,speed_b);
}

void actuate(float throttle, float steering){
  float speed_inc = 0;
  if(throttle > 0){
    speed_inc = (throttle - carSpeed) / 20;
  } else {
    speed_inc = (throttle - 1) / 10;
  }
  carSpeed = carSpeed + speed_inc;

  carSpeed = max(min(1.0, carSpeed), 0);

  float wheelDiff;
  if (carSpeed > 0.0){
  //if (carSpeed > 0.15){
    //wheelDiff = (1 - carSpeed)/0.85;
    wheelDiff = (1 - carSpeed);
  } else { 
    wheelDiff = 1;
  }
  float wheelDiffSteer = (0.5 + -0.3*wheelDiff) * steering;
  
  float rightWheel = carSpeed - wheelDiffSteer;
  float leftWheel = carSpeed + wheelDiffSteer;

  if(rightWheel>0.999){
    leftWheel = leftWheel - (rightWheel - 0.999);
  }
  
  if(leftWheel>0.999){
    rightWheel = rightWheel - (leftWheel - 0.999);
  }
  
  rightWheel = min(0.999, rightWheel);
  //rightWheel = rightWheel * 0.8;
  leftWheel = min(0.999, leftWheel);

  //speed_a = (uint8_t)(leftWheel * 256);
  //speed_b = (uint8_t)(rightWheel * 256);
  if(carSpeed<=0){
    speed_a = 0;
    speed_b = 0;
  }else{
    speed_b = (uint8_t)((leftWheel * 171) + 85);
    speed_a = (uint8_t)(((rightWheel * 171) + 85)*0.8);
  }
  
  //Serial.write(speed_a);
  //Serial.write(speed_b);
  
  analogWrite(ENA,speed_a);
  analogWrite(ENB,speed_b);
  //Serial.println((int)speed_a);
  //Serial.println((int)speed_b);
}

void flushBuffer(){
  while(Serial.available()>0){
    Serial.read();
  }
}

void setup() {
  // put your setup code here, to run once:
  //Serial.begin(115200);//open serial and set the baudrate
  Wire.begin();
  pinMode(IN1,OUTPUT);//left//before useing io pin, pin mode must be set first 
  pinMode(IN2,OUTPUT);
  pinMode(IN3,OUTPUT);//right
  pinMode(IN4,OUTPUT);
  pinMode(ENA,OUTPUT);//left
  pinMode(ENB,OUTPUT);//right

  digitalWrite(IN1,HIGH); //set IN1 hight level
  digitalWrite(IN2,LOW);  //set IN2 low level
  digitalWrite(IN3,HIGH);  //set IN3 low level
  digitalWrite(IN4,LOW); //set IN4 hight level

  Serial.begin(9600);
  //pinMode(10,INPUT);
  //pinMode(4,INPUT);
  //pinMode(2,INPUT);
  //myservo.attach(3);
  //myservo.write(80);
  //myservo.detach();

  while(Serial.available()<1){
    delay(1);
  }
  Serial.read();
  Serial.write(1);
  
}

void loop() {
  // put your main code here, to run repeatedly:
  
  char ir_buf[10];
  
  uint8_t count = 0;
  Wire.requestFrom(0x11, 10);
  while (Wire.available() ) { // slave may send less than requested
    char c = Wire.read(); // receive a byte as character
    ir_buf[count] = c;
    count += 1;
    
  }

  
  //Serial.print("Tick\n");
  bool ignore = false;
  int counter = 0;
  while(Serial.available()<1){
    if(counter < 500){
      counter++;
      delay(1); 
    }else if(counter == 500){
      //Serial.print("Safe stop\n");
      safeStop();
      ignore = true;
    }
  }
  //if(!ignore){
    uint8_t raspberry_buf[2];
    
    uint8_t i = 0;
  
    while(i < 2){
      if(Serial.available()>0){
        raspberry_buf[i] = Serial.read();
        i = i+1;
      }
    }
    //Serial.write(0);
    uint8_t send_speed = (uint8_t)(carSpeed*255);
    for(int j=0; j<10; j++){
      Serial.write(ir_buf[j]);
    }
    Serial.write(send_speed);
    float throttle = (float)(raspberry_buf[1]) / 128 - 1;
    float steering = (float)(raspberry_buf[0]) / 128 - 1;
    
    actuate(throttle, steering);
  //}
}
