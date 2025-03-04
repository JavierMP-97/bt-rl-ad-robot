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

int speed_a = 0;

int speed_b = 0;


int carSpeed = 175;

int Echo = A4;  
int Trig = A5; 

NewPing sonar(Trig, Echo);
int md = 0;

int wait=30;

void forward(){
  speed_a=carSpeed;
  speed_b=carSpeed;
  analogWrite(ENA, carSpeed);
  analogWrite(ENB, carSpeed);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  //Serial.println("go forward!");
}

void back(){
  analogWrite(ENA, carSpeed);
  analogWrite(ENB, carSpeed);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  //Serial.println("go back!");
}

void left(){
  analogWrite(ENA, carSpeed);
  analogWrite(ENB, carSpeed);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  //Serial.println("go left!");
}

void right(){
  analogWrite(ENA, carSpeed);
  analogWrite(ENB, carSpeed);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW); 
  //Serial.println("go right!");
} 

void turn_right(){
  if(speed_a>=speed_b){
    speed_b=speed_a;
  }else{
    speed_a=speed_b;
  }
  speed_b=speed_b*0.1;
  analogWrite(ENA,speed_a);
  analogWrite(ENB,speed_b);
}

void turn_left(){
  if(speed_a>=speed_b){
    speed_b=speed_a;
  }else{
    speed_a=speed_b;
  }
  speed_a=speed_a*0.1;
  analogWrite(ENA,speed_a);
  analogWrite(ENB,speed_b);
}

void stop(){
   digitalWrite(ENA, LOW);
   digitalWrite(ENB, LOW);
   //Serial.println("Stop!");
} 

//Ultrasonic distance measurement Sub function
int Distance_test() {
  digitalWrite(Trig, LOW);   
  delayMicroseconds(2);
  digitalWrite(Trig, HIGH);  
  delayMicroseconds(20);
  digitalWrite(Trig, LOW);   
  float Fdistance = pulseIn(Echo, HIGH);  
  Fdistance= Fdistance / 58;       
  return (int)Fdistance;
}

void setup(){
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
  delay(1000);
  forward();
}

void loop() {
  md = sonar.ping_cm();
  Serial.println(String(md));
  if(md==0){
    md=200;
  }

  if(md<20){
    stop();
    int i=0;
    carSpeed=100;
    while (i<wait&&md<20){
      delay(100);
      md = Distance_test();
      if(i==(wait-1)){
        
        left();
        while(!LT_L);
      }
      i++;
    }
    carSpeed=100;
  }else if(LT_M){
    if(LT_L){
      turn_right();
    }else if(LT_R){
      turn_left();
    }else{
      forward();
    }
  }
  else if(LT_R) { 
    right();
    while(LT_R);                             
  }   
  else if(LT_L) {
    left();
    while(LT_L);  
  }
  
  Serial.println(String(LT_L)+String(LT_M)+String(LT_R));
}
