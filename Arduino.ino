#include <Servo.h>

Servo servoMotor;
int emgPin = A0; 
int emgValue = 0;
int motorCommand = 0;

void setup() {
  servoMotor.attach(9);
  Serial.begin(9600); 
}

void loop() {
  emgValue = analogRead(emgPin);
  Serial.println(emgValue); 
  
  if (Serial.available() > 0) {
    motorCommand = Serial.read();
    servoMotor.write(motorCommand); 
  }

  delay(10);
}