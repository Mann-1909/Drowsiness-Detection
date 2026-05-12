#!/usr/bin/env python3
import time
import threading
import cv2
import numpy as np
import random
from flask import Flask, jsonify, Response

app = Flask(__name__)

# --- SHARED DATA (Matches exact structure of the real system) ---
sensor_data = {
    "alcohol_ppm": 0.0,
    "smoke_ppm": 0.0,
    "heart_rate": 75,
    "spo2": 98,                 # NEW
    "temperature_1": 24.5,
    "temperature_2": 36.5,
    "cpu_temp": 45.0,
    "status": "SAFE",
    "fatigue_score": 0.0,
    "latitude": 9.9312,   # Fake GPS (Kochi)
    "longitude": 76.2673,
    "is_night_mode": False,
    
    # NEW: MPU-6050 Physics Data
    "accel_x": 0.0,
    "accel_y": 0.0,
    "accel_z": 1.0, 
    "crash_detected": 0,
    
    # NEW: FSR Steering Wheel Data
    "grip_pressure": 85
}

# --- DUMMY VIDEO FEED ---
def generate_fake_video():
    while True:
        # Create a blank black/dark-gray image
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 30
        
        # Write the current status on the video feed
        text = f"SIMULATION: {sensor_data['status']}"
        color = (0, 255, 0)
        if sensor_data['status'] != "SAFE":
            color = (0, 0, 255)
            
        cv2.putText(frame, text, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(frame, f"Fatigue: {int(sensor_data['fatigue_score'])}%", (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Grip: {int(sensor_data['grip_pressure'])}%  |  SpO2: {int(sensor_data['spo2'])}%", (30, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        
        # Draw fake face bounding box
        cv2.rectangle(frame, (220, 50), (420, 250), (0, 255, 255), 2)
        
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if flag:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_fake_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/sensors', methods=['GET'])
def get_sensors():
    # Add slight random jitter to G-forces so the UI radar looks "alive"
    if sensor_data["crash_detected"] == 0 and sensor_data["status"] == "SAFE":
        sensor_data["accel_x"] = round(random.uniform(-0.05, 0.05), 2)
        sensor_data["accel_y"] = round(random.uniform(-0.05, 0.05), 2)
        sensor_data["accel_z"] = round(random.uniform(0.95, 1.05), 2)
        
    return jsonify(sensor_data)

# --- SIMULATION LOOP ---
def run_simulation():
    print("--- FAKE SENSOR SIMULATION STARTED ---")
    print("This will cycle through different danger scenarios every 10 seconds.")
    
    while True:
        # Phase 1: NORMAL / SAFE
        print("Scenario: NORMAL DRIVING")
        sensor_data.update({
            "status": "SAFE", "alcohol_ppm": 5.0, "smoke_ppm": 10.0, 
            "fatigue_score": 3.0, "heart_rate": 72, "spo2": 99, 
            "cpu_temp": 61.2, "temperature_1": 24.5, "temperature_2": 36.5,
            "grip_pressure": 88, "crash_detected": 0,
            "accel_x": 0.0, "accel_y": 0.0, "accel_z": 1.0
        })
        time.sleep(10)
        
        # Phase 2: HANDS OFF WHEEL
        print("Scenario: HANDS OFF WHEEL")
        sensor_data.update({
            "status": "HANDS OFF WHEEL", 
            "grip_pressure": 0, # FSR triggers
            "heart_rate": 75, "fatigue_score": 10.0
        })
        time.sleep(10)
        
        # Phase 3: DROWSY WARNING
        print("Scenario: DROWSY WARNING")
        sensor_data.update({
            "status": "DROWSY WARNING", "fatigue_score": 53.0, 
            "heart_rate": 65, "spo2": 96, "grip_pressure": 60,
            "temperature_1": 28.5 # Cabin getting hot
        })
        time.sleep(10)
        
        # Phase 4: CRITICAL FATIGUE
        print("Scenario: CRITICAL FATIGUE")
        sensor_data.update({
            "status": "CRITICAL RISK", "fatigue_score": 88.0, 
            "heart_rate": 58, "spo2": 94, "grip_pressure": 20
        })
        time.sleep(10)
        
        # Phase 5: AGGRESSIVE DRIVING (Hard Braking / Cornering)
        print("Scenario: AGGRESSIVE DRIVING (RADAR TEST)")
        sensor_data.update({
            "status": "SAFE", "fatigue_score": 15.0, "grip_pressure": 95,
            "accel_x": 0.9,   # Hard right swerve
            "accel_y": -1.2,  # Hard braking
            "accel_z": 1.1
        })
        time.sleep(10) # Shorter time for swerve
        
        # Phase 6: THE CRASH
        print("Scenario: CRASH DETECTED")
        sensor_data.update({
            "status": "CRASH DETECTED", "crash_detected": 1,
            "grip_pressure": 0, "heart_rate": 110, "spo2": 98,
            "accel_x": 3.5, "accel_y": 4.2, "accel_z": -1.5 # Massive G-Spike
        })
        time.sleep(10)
        
        # Phase 7: ALCOHOL DETECTED
        print("Scenario: ALCOHOL SPIKE")
        sensor_data.update({
            "status": "ALCOHOL DANGER", "crash_detected": 0,
            "alcohol_ppm": 125.0, "smoke_ppm": 12.0, 
            "fatigue_score": 5.0, "grip_pressure": 70,
            "accel_x": 0.0, "accel_y": 0.0, "accel_z": 1.0
        })
        time.sleep(10)
        
        # Phase 8: SMOKE/FIRE DETECTED
        print("Scenario: SMOKE SPIKE")
        sensor_data.update({
            "status": "SMOKE DANGER", "alcohol_ppm": 0.0,
            "smoke_ppm": 450.0, "heart_rate": 95, "cpu_temp": 85.0,
            "temperature_1": 42.0 # Fire making cabin hot
        })
        time.sleep(10)

        # Phase 9: BODY TEMPERATURE SPIKE
        print("Scenario: BODY TEMPERATURE SPIKE")
        sensor_data.update({
            "status": "BODY TEMPERATURE SPIKE", "smoke_ppm": 15.0,
            "temperature_2": 39.5, # Fever
            "heart_rate": 105, "temperature_1": 25.0
        })
        time.sleep(10)

        print("Scenario: CABIN TEMPERATURE SPIKE")
        sensor_data.update({
            "status": "CABIN TEMPERATURE SPIKE", "smoke_ppm": 15.0,
            "temperature_1": 39.5, # Fever
            "heart_rate": 105, "temperature_2": 25.0
        })
        time.sleep(10)

if __name__ == '__main__':
    # Start the simulation thread
    sim_thread = threading.Thread(target=run_simulation)
    sim_thread.daemon = True
    sim_thread.start()
    
    # Run server
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)