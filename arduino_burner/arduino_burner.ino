#define USE_ARDUINO_INTERRUPTS true
#include <OneWire.h>
#include <DallasTemperature.h>
#include <MQUnifiedsensor.h>
#include <Wire.h> 

// --- MAX30102 LIBRARIES ---
#include "MAX30105.h"
#include "heartRate.h"

// --- PIN DEFINITIONS ---
const int PIN_MQ3     = A0; // Alcohol
const int PIN_FSR     = A1; // NEW: Force Sensitive Resistor on Steering Wheel
const int PIN_MQ2     = A2; // Smoke/LPG
const int PIN_TEMP_1  = 2;  
const int PIN_TEMP_2  = 3;  
const int PIN_BUZZER  = 8;  // Arduino Buzzer Pin

// --- MQ SENSOR CONFIG ---
#define Board "Arduino UNO"
#define Voltage_Resolution 5
#define ADC_Bit_Resolution 10
#define Type_MQ3 "MQ-3"
#define Type_MQ2 "MQ-2"

MQUnifiedsensor MQ3(Board, Voltage_Resolution, ADC_Bit_Resolution, PIN_MQ3, Type_MQ3);
MQUnifiedsensor MQ2(Board, Voltage_Resolution, ADC_Bit_Resolution, PIN_MQ2, Type_MQ2);

// --- TEMP SENSOR CONFIG ---
OneWire oneWire1(PIN_TEMP_1);
OneWire oneWire2(PIN_TEMP_2);
DallasTemperature sensor1(&oneWire1);
DallasTemperature sensor2(&oneWire2);
float tempC1 = 0.0;
float tempC2 = 0.0;

// --- MPU-6050 CONFIG ---
const int MPU_ADDR = 0x68; 

// --- MAX30102 CONFIG (BPM & SpO2) ---
MAX30105 particleSensor;
const byte RATE_SIZE = 4;
byte rates[RATE_SIZE]; 
byte rateSpot = 0;
long lastBeat = 0; 
float beatsPerMinute;
int beatAvg = 0; 

// SpO2 Variables
double aveRed = 0.0;
double aveIr = 0.0;
double sumRedRMS = 0.0;
double sumIrRMS = 0.0;
int validSpO2 = 0; 
int spO2Samples = 0;

// --- TIMING VARIABLES ---
unsigned long lastTempRequest = 0;
unsigned long lastSerialTransmit = 0;
const int TEMP_UPDATE_INTERVAL = 1000;
const int SERIAL_UPDATE_INTERVAL = 100; // Sends data to Pi 10 times a second

void setup() {
  Serial.begin(9600); 
  Wire.begin();

  pinMode(PIN_BUZZER, OUTPUT);
  digitalWrite(PIN_BUZZER, HIGH); // Assuming active LOW buzzer
  pinMode(PIN_FSR, INPUT);

  // 1. Setup Temp Sensors
  sensor1.begin(); sensor1.setWaitForConversion(false);
  sensor2.begin(); sensor2.setWaitForConversion(false);
  sensor1.requestTemperatures();
  sensor2.requestTemperatures();

  // 2. Setup MQ Sensors
  MQ3.setRegressionMethod(1); 
  MQ3.setA(0.3934); MQ3.setB(-1.504); 
  MQ3.init();
  MQ2.setRegressionMethod(1); 
  MQ2.setA(574.25); MQ2.setB(-2.222); 
  MQ2.init();
  MQ3.setR0(400); 
  MQ2.setR0(10); 

  // 3. Setup MPU-6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); 
  Wire.write(0);    
  Wire.endTransmission(true);

  // 4. Setup MAX30102 Heart Rate & Oximeter Sensor
  if (particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    particleSensor.setup(); 
    particleSensor.setPulseAmplitudeRed(0x3F); 
    particleSensor.setPulseAmplitudeIR(0x3F);  
    particleSensor.setPulseAmplitudeGreen(0);  
  }
}

void loop() {
  // --- CHECK FOR BUZZER COMMAND FROM PI ---
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == '1') {
      digitalWrite(PIN_BUZZER, LOW); 
    } else if (cmd == '0') {
      digitalWrite(PIN_BUZZER, HIGH);  
    }
  }

  // --- READ HEART RATE & SpO2 ---
  particleSensor.check(); 
  
  while (particleSensor.available()) {
    long irValue = particleSensor.getFIFOIR();
    long redValue = particleSensor.getFIFORed();
    particleSensor.nextSample(); 

    if (checkForBeat(irValue) == true) {
      long delta = millis() - lastBeat;
      lastBeat = millis();
      beatsPerMinute = 60 / (delta / 1000.0);
      if (beatsPerMinute < 255 && beatsPerMinute > 20) {
        rates[rateSpot++] = (byte)beatsPerMinute; 
        rateSpot %= RATE_SIZE; 
        beatAvg = 0;
        for (byte x = 0 ; x < RATE_SIZE ; x++) beatAvg += rates[x];
        beatAvg /= RATE_SIZE;
      }
    }
    
    if (irValue < 20000) {
      beatAvg = 0;
      validSpO2 = 0; 
    } else {
      if (aveRed == 0.0) aveRed = redValue; 
      if (aveIr == 0.0) aveIr = irValue;
      aveRed = (aveRed * 0.95) + ((double)redValue * 0.05);
      aveIr = (aveIr * 0.95) + ((double)irValue * 0.05);
      double redAC = (double)redValue - aveRed;
      double irAC = (double)irValue - aveIr;
      sumRedRMS += redAC * redAC;
      sumIrRMS += irAC * irAC;
      spO2Samples++;

      if (spO2Samples >= 100) { 
          double ratio = sqrt(sumRedRMS / sumIrRMS) * (aveIr / aveRed);
          int calculatedSpO2 = 110 - (25 * ratio);
          if (calculatedSpO2 > 100) calculatedSpO2 = 100;
          if (calculatedSpO2 < 70) calculatedSpO2 = 70;
          validSpO2 = calculatedSpO2;
          sumRedRMS = 0.0; sumIrRMS = 0.0; spO2Samples = 0;
      }
    }
  }

  // --- READ TEMPERATURES ---
  if (millis() - lastTempRequest >= TEMP_UPDATE_INTERVAL) {
    tempC1 = sensor1.getTempCByIndex(0);
    tempC2 = sensor2.getTempCByIndex(0);
    if (tempC1 <= -100.0) tempC1 = 0.0;
    if (tempC2 <= -100.0) tempC2 = 0.0;
    sensor1.requestTemperatures(); 
    sensor2.requestTemperatures(); 
    lastTempRequest = millis();
  }

  // --- SEND DATA TO PI ---
  if (millis() - lastSerialTransmit >= SERIAL_UPDATE_INTERVAL) {
    
    MQ3.update(); float alcoholPPM = MQ3.readSensor(); 
    MQ2.update(); float smokePPM = MQ2.readSensor();

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B); 
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 6, true); 
    int16_t AcX = Wire.read()<<8 | Wire.read();  
    int16_t AcY = Wire.read()<<8 | Wire.read();  
    int16_t AcZ = Wire.read()<<8 | Wire.read();  
    float gX = AcX / 16384.0;
    float gY = AcY / 16384.0;
    float gZ = AcZ / 16384.0;

    // --- NEW: READ FSR GRIP PRESSURE ---
    int fsrRaw = analogRead(PIN_FSR);
    // Map the analog reading to a 0-100% scale
    // Calibrated for a 2.2k Ohm pull-down resistor (Max raw value is ~500 instead of 1023)
    int gripPercent = map(fsrRaw, 0, 500, 0, 100);
    
    // Constrain to ensure we don't send negative numbers or exceed 100%
    if (gripPercent < 0) gripPercent = 0;
    if (gripPercent > 100) gripPercent = 100;

    // Send 10 values to Raspberry Pi
    Serial.print(alcoholPPM); Serial.print(",");
    Serial.print(smokePPM);   Serial.print(",");
    Serial.print(beatAvg);    Serial.print(",");
    Serial.print(validSpO2);  Serial.print(",");
    Serial.print(tempC1);     Serial.print(",");
    Serial.print(tempC2);     Serial.print(",");
    Serial.print(gX);         Serial.print(",");
    Serial.print(gY);         Serial.print(",");
    Serial.print(gZ);         Serial.print(",");
    Serial.println(gripPercent); // 10th value

    lastSerialTransmit = millis(); 
  }
}