#!/usr/bin/env python3
"""
Driver Safety System - Main Edge Node Application
-------------------------------------------------
This script acts as the core edge-computing node for an intelligent Driver Assistance System.
It runs a multi-threaded Flask application on a Raspberry Pi that performs:
1. Real-time Computer Vision (Face tracking, eye-closure detection, head pose estimation)
2. Multi-sensor Data Fusion (Alcohol, Smoke, Heart Rate, Temperatures) via Arduino serial bridge.
3. Fuzzy Logic Inference for risk assessment and linguistic categorization.
4. Intelligent Alerts (Buzzer patterns and Espeak voice warnings).
5. Data Logging & GPS tracking (Black Box Recorder).
6. Local MJPEG streaming via HTTP for dashboard monitoring.
"""

import time
import os
import cv2
import numpy as np
import threading
import math
import serial
import csv
import subprocess
from datetime import datetime
from flask import Flask, jsonify, Response
from picamera2 import Picamera2

# --- FLASK APP SETUP ---
app = Flask(__name__)

# --- GLOBAL VARIABLES ---
output_frame = np.zeros((480, 640, 3), dtype=np.uint8) 
frame_lock = threading.Lock() 
arduino_serial_conn = None 

# Shared Data Dictionary acting as the central state for the entire system
sensor_data = {
    "alcohol_ppm": 0.0,
    "smoke_ppm": 0.0,
    "heart_rate": 0,
    "temperature_1": 0.0, 
    "temperature_2": 0.0, 
    "cpu_temp": 0.0,
    "status": "BOOTING",
    "fatigue_score": 0.0,
    "latitude": 9.93120,
    "longitude": 76.26730,
    "is_night_mode": False,
    
    # MPU-6050 Physics Data
    "accel_x": 0.0,
    "accel_y": 0.0,
    "accel_z": 1.0, # Gravity defaults to 1G pointing down
    "crash_detected": 0,
    "crash_time": 0.0, # NEW: Tracks the exact time of impact
    
    # Fuzzy Linguistic Descriptors for Dashboard UI
    "fuzzy_alcohol": "Unknown",
    "fuzzy_smoke": "Unknown",
    "fuzzy_heart_rate": "Unknown",
    "fuzzy_body_temp": "Unknown",
    "fuzzy_cabin_temp": "Unknown"
}

# Create Evidence Folder to store snapshots during critical risk events
EVIDENCE_PATH = "/home/pi/Desktop/DDD/evidence"
if not os.path.exists(EVIDENCE_PATH): os.makedirs(EVIDENCE_PATH)

# --- CONFIGURATION ---
CASCADE_PATH = "/home/pi/Desktop/DDD/haarcascade/"
TFLITE_MODEL_PATH = "/home/pi/Desktop/DDD/my_model.tflite"

DRIVER_ZONE_X_MAX = 0.75 
ENABLE_SKELETON = True
FRAME_SKIP = 1 

CPU_WARN_THRESHOLD = 83.0 
CRASH_THRESHOLD_G = 3.0 # Crash logic threshold (calculated by Pi)
EYE_CONFIDENCE_THRESHOLD = 0.5 
MAR_THRESHOLD = 0.5        
PITCH_THRESHOLD = 160     
ROLL_THRESHOLD = 38        
YAW_DISTRACTION_THRESHOLD = 40 

SCORE_DECAY = 3          
SCORE_EYES_CLOSED = 4.0    
SCORE_YAWN = 2.0           
SCORE_DISTRACTED = 4.0   

# --- LOAD MODELS ---
faceCascade = cv2.CascadeClassifier(os.path.join(CASCADE_PATH, "haarcascade_frontalface_default.xml"))
eyeCascade  = cv2.CascadeClassifier(os.path.join(CASCADE_PATH, "haarcascade_eye.xml"))

interpreter = None
input_details = None
output_details = None
using_tflite = False
in_height = in_width = 224

if os.path.exists(TFLITE_MODEL_PATH):
    try:
        from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
        interpreter = TFLiteInterpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        in_height = int(input_details[0]['shape'][1])
        in_width  = int(input_details[0]['shape'][2])
        using_tflite = True
        print(f"Loaded TFLite Model: {in_width}x{in_height}")
    except Exception as e:
        print("Failed to load TFLite model:", e)

# --- MEDIA PIPE SETUP ---
USE_MEDIAPIPE = False
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    USE_MEDIAPIPE = True
    print("MediaPipe Loaded Successfully.")
except Exception as e:
    USE_MEDIAPIPE = False
    print(f"MediaPipe Error: {e}")

# --- HELPERS ---
def speak_warning(text):
    try:
        subprocess.Popen(['espeak', '-s', '150', text], stderr=subprocess.DEVNULL) 
    except: pass

def get_cpu_temperature():
    try:
        res = subprocess.check_output(['vcgencmd', 'measure_temp']).decode('utf-8')
        return float(res.replace("temp=","").replace("'C\n",""))
    except: return 0.0

def save_evidence_snapshot(frame, reason):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{EVIDENCE_PATH}/ALERT_{reason}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)

def adjust_gamma(image, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def calculate_mar(face_landmarks, w, h):
    try:
        top = (face_landmarks.landmark[13].x * w, face_landmarks.landmark[13].y * h)
        bot = (face_landmarks.landmark[14].x * w, face_landmarks.landmark[14].y * h)
        left = (face_landmarks.landmark[78].x * w, face_landmarks.landmark[78].y * h)
        right = (face_landmarks.landmark[308].x * w, face_landmarks.landmark[308].y * h)
        vertical_dist = math.hypot(top[0]-bot[0], top[1]-bot[1])
        horizontal_dist = math.hypot(left[0]-right[0], left[1]-right[1])
        if horizontal_dist == 0: return 0.0
        return vertical_dist / horizontal_dist
    except: return 0.0


# ==============================================================================
# --- ADVANCED MULTI-VARIABLE FUZZY LOGIC ENGINE ---
# ==============================================================================
class FuzzyLogicEngine:
    @staticmethod
    def trimf(x, a, b, c):
        if x <= a or x >= c: return 0.0
        if a < x <= b: return (x - a) / (b - a) if b != a else 1.0
        if b < x < c: return (c - x) / (c - b) if c != b else 1.0
        return 0.0

    @staticmethod
    def trapmf(x, a, b, c, d):
        if x <= a or x >= d: return 0.0
        if a < x <= b: return (x - a) / (b - a) if b != a else 1.0
        if b < x <= c: return 1.0
        if c < x < d: return (d - x) / (d - c) if d != c else 1.0
        return 0.0

    @staticmethod
    def get_linguistic_descriptor(memberships):
        memberships.sort(key=lambda x: x[0], reverse=True)
        top_val, top_label = memberships[0]
        if top_val == 0: return "Unknown"
        if top_val >= 0.7: return top_label 
        elif 0.3 <= top_val < 0.7:
            if "Normal" in top_label or "Safe" in top_label:
                return "Near " + top_label
            return "Slightly " + top_label
        else: return "Borderline " + top_label

    @classmethod
    def evaluate_all(cls, cam_fatigue, alc_ppm, smoke_ppm, heart_rate, body_temp, cabin_temp):
        cam_awake = cls.trapmf(cam_fatigue, 0, 0, 20, 40)
        cam_drowsy = cls.trimf(cam_fatigue, 30, 50, 70)
        cam_sleep = cls.trapmf(cam_fatigue, 60, 80, 100, 100)
        
        alc_safe = cls.trapmf(alc_ppm, 0, 0, 20, 50)
        alc_warn = cls.trimf(alc_ppm, 30, 60, 90)
        alc_dang = cls.trapmf(alc_ppm, 70, 100, 1000, 1000)
        
        smk_safe = cls.trapmf(smoke_ppm, 0, 0, 150, 250)
        smk_warn = cls.trimf(smoke_ppm, 200, 300, 400)
        smk_dang = cls.trapmf(smoke_ppm, 350, 450, 1000, 1000)
        
        hr_low  = cls.trapmf(heart_rate, 0, 0, 30, 45)
        hr_norm = cls.trapmf(heart_rate, 40, 60, 250, 250)
        hr_high = cls.trapmf(heart_rate, 250, 260, 300, 300)
        if heart_rate == 0: hr_low = hr_norm = hr_high = 0.0
        
        bt_low  = cls.trapmf(body_temp, 0, 0, 30.0, 31.5)
        bt_norm = cls.trimf(body_temp, 31.0, 34.5, 37.8)
        bt_high = cls.trapmf(body_temp, 37.5, 38.5, 50.0, 50.0)
        if body_temp == 0: bt_low = bt_norm = bt_high = 0.0

        ct_cold = cls.trapmf(cabin_temp, 0, 0, 15.0, 22.0)
        ct_norm = cls.trimf(cabin_temp, 18.0, 24.0, 28.0)
        ct_hot  = cls.trapmf(cabin_temp, 26.0, 30.0, 60.0, 60.0)
        if cabin_temp == 0: ct_cold = ct_norm = ct_hot = 0.0

        fuzzy_texts = {
            "alcohol": cls.get_linguistic_descriptor([(alc_safe, "Safe"), (alc_warn, "Elevated"), (alc_dang, "Danger")]),
            "smoke": cls.get_linguistic_descriptor([(smk_safe, "Safe"), (smk_warn, "Elevated"), (smk_dang, "Danger")]),
            "heart_rate": cls.get_linguistic_descriptor([(hr_low, "Low"), (hr_norm, "Normal"), (hr_high, "High")]),
            "body_temp": cls.get_linguistic_descriptor([(bt_low, "Low"), (bt_norm, "Normal"), (bt_high, "Fever")]),
            "cabin_temp": cls.get_linguistic_descriptor([(ct_cold, "Cold"), (ct_norm, "Optimal"), (ct_hot, "Hot")])
        }

        rule_crit = max(cam_sleep, alc_dang, smk_dang, hr_high, hr_low)
        rule_med = max(cam_drowsy, alc_warn, smk_warn, bt_high, ct_hot)
        rule_low = max(cam_awake, alc_safe, smk_safe, hr_norm, bt_norm)

        numerator = (rule_low * 10) + (rule_med * 60) + (rule_crit * 95)
        denominator = rule_low + rule_med + rule_crit
        
        final_score = 0.0
        if denominator > 0: 
            final_score = numerator / denominator
            
        return max(0.0, min(100.0, final_score)), fuzzy_texts

current_camera_score = 0.0
def calculate_raw_camera_fatigue(face_detected, is_closed, is_yawning, is_distracted):
    global current_camera_score
    if not face_detected: 
        current_camera_score = 0.0
        return 0.0
    elif is_closed: current_camera_score += SCORE_EYES_CLOSED
    elif is_yawning: current_camera_score += SCORE_YAWN
    elif is_distracted: current_camera_score += SCORE_DISTRACTED
    else: current_camera_score -= SCORE_DECAY
    
    current_camera_score = max(0.0, min(100.0, current_camera_score))
    return current_camera_score
# ==============================================================================

# --- THREADS ---
def run_arduino_bridge():
    global sensor_data, arduino_serial_conn
    print("--- Scanning for Arduino... ---")
    arduino = None
    
    ports = ['/dev/ttyACM0', '/dev/ttyUSB0', '/dev/ttyACM1', '/dev/ttyUSB1']
    for p in ports:
        try:
            arduino = serial.Serial(p, 9600, timeout=1)
            arduino_serial_conn = arduino 
            print(f"SUCCESS: Arduino Connected on {p}")
            try:
                arduino.write(b'1')
                time.sleep(1)
                arduino.write(b'0')
            except: pass
            break
        except: pass
        
    if arduino is None:
        print("ERROR: Arduino not found! Check USB connection.")
        return

    while True:
        try:
            if arduino.in_waiting > 0:
                line = arduino.readline().decode('utf-8').strip()
                if not line: continue
                
                parts = line.split(',')
                # Expecting 9 values now: alc, smk, bpm, spo2, temp1, temp2, gX, gY, gZ
                if len(parts) >= 9:
                    sensor_data["alcohol_ppm"] = float(parts[0])
                    sensor_data["smoke_ppm"] = float(parts[1])
                    sensor_data["heart_rate"] = int(parts[2])
                    sensor_data["spo2"] = int(parts[3]) # NEW: Pulse Oximeter Reading
                    sensor_data["temperature_1"] = float(parts[4])
                    sensor_data["temperature_2"] = float(parts[5])
                    
                    gX = float(parts[6])
                    gY = float(parts[7])
                    gZ = float(parts[8])
                    
                    sensor_data["accel_x"] = gX
                    sensor_data["accel_y"] = gY
                    sensor_data["accel_z"] = gZ
                    
                    # --- CRASH LOGIC EXECUTED ON RASPBERRY PI WITH 10 SEC SUSTAIN ---
                    g_magnitude = math.sqrt(gX**2 + gY**2 + gZ**2)
                    if g_magnitude > CRASH_THRESHOLD_G:
                        sensor_data["crash_detected"] = 1
                        sensor_data["crash_time"] = time.time() # Start/Reset the 10 sec timer
                    else:
                        # Only clear the flag if 10 seconds have passed since the impact
                        if time.time() - sensor_data.get("crash_time", 0.0) > 10.0:
                            sensor_data["crash_detected"] = 0
                        
        except Exception as e:
            time.sleep(0.5)

def run_data_logger():
    filename = datetime.now().strftime("/home/pi/Desktop/DDD/tripData/trip_log_%Y%m%d_%H%M%S.csv")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Status", "FatigueScore", "BPM", "Alcohol_PPM", "Smoke_PPM", "TempBody", "Gx", "Gy", "Gz", "CrashFlag"])
    while True:
        try:
            sensor_data["cpu_temp"] = get_cpu_temperature()
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                timestamp = datetime.now().strftime("%H:%M:%S")
                writer.writerow([timestamp, sensor_data["status"], round(sensor_data["fatigue_score"], 1),
                                sensor_data["heart_rate"], 
                                sensor_data["alcohol_ppm"], 
                                sensor_data["smoke_ppm"], sensor_data["temperature_2"], 
                                round(sensor_data["accel_x"], 2), round(sensor_data["accel_y"], 2), round(sensor_data["accel_z"], 2),
                                sensor_data["crash_detected"]])
            time.sleep(1.0)
        except: time.sleep(1.0)

def control_buzzer(state):
    global arduino_serial_conn
    if arduino_serial_conn is not None and arduino_serial_conn.is_open:
        try:
            arduino_serial_conn.write(b'1' if state else b'0')
        except: pass

def run_buzzer_control():
    global sensor_data
    last_spoken_time = 0
    last_snapshot_time = 0
    warning_vitals_start = None
    slight_vitals_start = None

    while True:
        score = sensor_data["fatigue_score"]
        status = sensor_data["status"]
        cpu = sensor_data["cpu_temp"]
        crash = sensor_data["crash_detected"]
        current_time = time.time()

        if status == "CALIBRATING":
            time.sleep(1)
            continue
        
        f_alc = sensor_data["fuzzy_alcohol"]
        f_smk = sensor_data["fuzzy_smoke"]
        f_hr = sensor_data["fuzzy_heart_rate"]
        f_bt = sensor_data["fuzzy_body_temp"]

        # --- 1. CLASSIFY SEVERITY ---
        is_critical = False
        raw_warning = False
        raw_slight = False

        if crash == 1 or score > 80 or "Danger" in f_alc or "Danger" in f_smk or cpu > CPU_WARN_THRESHOLD:
            is_critical = True
        elif score > 50 or "Elevated" in f_alc or "Elevated" in f_smk or "Fever" in f_bt or f_hr in ["High", "Low", "Borderline High", "Borderline Low"]:
            raw_warning = True
        elif "Slightly" in f_hr or "Slightly" in f_bt or "Slightly" in f_alc or "Slightly" in f_smk:
            raw_slight = True

        # DEBOUNCE TIMERS
        is_warning = False
        is_slight = False

        if raw_warning and not is_critical:
            if warning_vitals_start is None: warning_vitals_start = current_time
            elif current_time - warning_vitals_start > 2.0: is_warning = True
        else: warning_vitals_start = None 

        if raw_slight and not is_critical and not raw_warning:
            if slight_vitals_start is None: slight_vitals_start = current_time
            elif current_time - slight_vitals_start > 2.0: is_slight = True
        else: slight_vitals_start = None 

        # --- 3. EXECUTE BUZZER PATTERNS & VOICE ALERTS ---
        if is_critical:
            if current_time - last_spoken_time > 5:
                if crash == 1: speak_warning("CRASH DETECTED. SOS SENT.")
                elif cpu > CPU_WARN_THRESHOLD: speak_warning("Warning. System Overheating.")
                elif "Danger" in f_alc: speak_warning("Alcohol Danger. Pull over.")
                elif "Danger" in f_smk: speak_warning("Fire Hazard Detected.")
                else: speak_warning("Critical Risk Detected.")
                last_spoken_time = current_time
                
            if (crash == 1 or score > 80) and (current_time - last_snapshot_time > 5):
                with frame_lock:
                    if output_frame is not None: 
                        save_evidence_snapshot(output_frame, "CRASH" if crash == 1 else "CRITICAL_RISK")
                last_snapshot_time = current_time

            for _ in range(3):
                control_buzzer(True); time.sleep(0.1); control_buzzer(False); time.sleep(0.1)
            time.sleep(0.5)

        elif is_warning:
            if current_time - last_spoken_time > 10:
                speak_warning("Warning. Elevated risk detected.")
                last_spoken_time = current_time

            for _ in range(2):
                control_buzzer(True); time.sleep(0.2); control_buzzer(False); time.sleep(0.2)
            time.sleep(1.0)
            
        elif is_slight:
            if current_time - last_spoken_time > 20:
                speak_warning("Vitals fluctuating.")
                last_spoken_time = current_time

            control_buzzer(True); time.sleep(0.5); control_buzzer(False); time.sleep(2.0) 
            
        else:
            control_buzzer(False); time.sleep(0.1)

def run_gps_system():
    global sensor_data
    try:
        gps = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)
        while True:
            try:
                line = gps.readline().decode('ascii', errors='ignore').strip()
                if line.startswith("$GPGGA"):
                    parts = line.split(',')
                    if len(parts) > 5 and parts[2] != '' and parts[4] != '':
                        lat_raw = float(parts[2])
                        lon_raw = float(parts[4])
                        lat_deg = int(lat_raw / 100)
                        lat_min = lat_raw - (lat_deg * 100)
                        lat = lat_deg + (lat_min / 60)
                        if parts[3] == 'S': lat = -lat
                        
                        lon_deg = int(lon_raw / 100)
                        lon_min = lon_raw - (lon_deg * 100)
                        lon = lon_deg + (lon_min / 60)
                        if parts[5] == 'W': lon = -lon
                        
                        sensor_data["latitude"] = lat
                        sensor_data["longitude"] = lon
            except: pass
    except: pass

def predict_eye_status(eye_roi):
    if eye_roi.size == 0: return 1.0
    try:
        resized = cv2.resize(eye_roi, (in_width, in_height))
        input_tensor = resized.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)
        if using_tflite and interpreter is not None:
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index'])
            return float(np.array(pred).flatten()[0]) 
    except: pass
    return 1.0

font = cv2.FONT_HERSHEY_SIMPLEX
def draw_hud(frame, pitch, roll, yaw, eye_status, score, yawn_status, calibration_progress=None):
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1) 
    
    if calibration_progress is not None:
        bar_w = int((calibration_progress / 100.0) * w)
        cv2.rectangle(overlay, (0, h//2 - 20), (bar_w, h//2 + 20), (0, 255, 255), -1)
        cv2.putText(frame, "CALIBRATING MULTI-SENSORS...", (w//2 - 150, h//2 + 10), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return

    bar_width = int((score / 100.0) * w)
    bar_color = (0, 255, 0)
    if score > 50: bar_color = (0, 255, 255)
    if score > 80: bar_color = (0, 0, 255)
    
    if sensor_data.get("crash_detected", 0) == 1: bar_color = (0, 0, 255)
    
    cv2.rectangle(overlay, (0, h - 20), (bar_width if sensor_data.get("crash_detected", 0) == 0 else w, h), bar_color, -1)

    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, f"Pitch: {int(pitch)} | Roll: {int(roll)} | Yaw: {int(yaw)}", (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    eye_color = (0, 0, 255) if eye_status == "Closed" else (0, 255, 0)
    cv2.putText(frame, f"Eyes: {eye_status}", (10, 45), font, 0.5, eye_color, 2, cv2.LINE_AA)
    if yawn_status: cv2.putText(frame, "YAWN!", (140, 45), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.putText(frame, f"HR: {sensor_data['fuzzy_heart_rate']}", (w - 220, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Alc: {sensor_data['fuzzy_alcohol']}", (w - 220, 45), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Smk: {sensor_data['fuzzy_smoke']}", (w - 220, 70), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ADDED: Small text in corner for 10 seconds during crash
    if sensor_data.get("crash_detected", 0) == 1:
        cv2.putText(frame, "STATUS: CRASH DETECTED!", (w - 220, 95), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Fatigue Risk: {int(score)}%", (10, 70), font, 0.7, bar_color, 2, cv2.LINE_AA)
    
    # Center Screen Alerts (Removed the massive crash text)
    if score > 80: cv2.putText(frame, "CRITICAL FATIGUE!", (w//2 - 170, h//2), font, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    elif score > 50: cv2.putText(frame, "WARNING: Drowsy", (w//2 - 160, h//2), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

# --- MAIN DETECTION LOOP ---
def run_detection_system():
    global output_frame, sensor_data
    
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480), "format": "YUV420"})
    picam2.configure(config)
    picam2.start()
    
    if USE_MEDIAPIPE:
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=4, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
    landmark_ids_pose = [1, 152, 33, 263, 61, 291]
    model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)], dtype=np.float64)

    print("--- Starting Calibration ---")
    speak_warning("System starting. Calibrating.")
    sensor_data["status"] = "CALIBRATING"
    
    calib_frames = 0
    total_calib_frames = 80
    
    while calib_frames < total_calib_frames:
        yuv = picam2.capture_array()
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        draw_hud(frame, 0,0,0, "Open", 0, False, calibration_progress=(calib_frames/total_calib_frames)*100)
        with frame_lock: output_frame = frame.copy()
        calib_frames += 1
        time.sleep(0.1)
    
    speak_warning("Calibration complete.")
    sensor_data["status"] = "SAFE"
    frame_count = 0 

    while True:
        try:
            yuv = picam2.capture_array()
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            frame = adjust_gamma(frame, gamma=1.2)
            h_img, w_img = frame.shape[:2]
            
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                draw_hud(frame, sensor_data.get("last_pitch", 0), sensor_data.get("last_roll", 0), sensor_data.get("last_yaw", 0), 
                         sensor_data.get("last_eyes", "Open"), sensor_data["fatigue_score"], False)
                with frame_lock: output_frame = frame.copy()
                time.sleep(0.01)
                continue

            status_text = "No Face"
            pitch = roll = yaw = 0.0
            face_detected = False
            is_yawning = False
            best_face = None
            
            if USE_MEDIAPIPE:
                rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if ENABLE_SKELETON:
                    pose_results = pose.process(rgb_small)
                    if pose_results.pose_landmarks:
                        landmarks = pose_results.pose_landmarks.landmark
                        if landmarks[0].x <= DRIVER_ZONE_X_MAX:
                            def to_coords(lm): return int(lm.x * w_img), int(lm.y * h_img)
                            try:
                                l_shoulder = to_coords(landmarks[11])
                                r_shoulder = to_coords(landmarks[12])
                                nose = to_coords(landmarks[0])
                                cv2.line(frame, l_shoulder, r_shoulder, (0, 255, 255), 2)
                                cv2.line(frame, l_shoulder, (l_shoulder[0], h_img), (0, 255, 255), 2)
                                cv2.line(frame, r_shoulder, (r_shoulder[0], h_img), (0, 255, 255), 2)
                                mid_shoulder = ((l_shoulder[0] + r_shoulder[0]) // 2, (l_shoulder[1] + r_shoulder[1]) // 2)
                                cv2.line(frame, mid_shoulder, nose, (0, 255, 255), 2)
                                cv2.circle(frame, l_shoulder, 5, (0, 0, 255), -1)
                                cv2.circle(frame, r_shoulder, 5, (0, 0, 255), -1)
                                cv2.circle(frame, mid_shoulder, 5, (0, 255, 0), -1)
                            except: pass

                results = face_mesh.process(rgb_small)
                
                max_area = 0
                if results.multi_face_landmarks:
                    for face_lms in results.multi_face_landmarks:
                        centroid_x = face_lms.landmark[1].x
                        if centroid_x > DRIVER_ZONE_X_MAX:
                            px, py = int(centroid_x * w_img), int(face_lms.landmark[1].y * h_img)
                            cv2.circle(frame, (px, py), 15, (100, 100, 100), -1)
                            continue 
                        
                        xs = [lm.x for lm in face_lms.landmark]
                        ys = [lm.y for lm in face_lms.landmark]
                        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                        
                        if area > max_area:
                            max_area = area
                            best_face = face_lms

                if best_face:
                    face_detected = True
                    face_landmarks = best_face
                    
                    mar = calculate_mar(face_landmarks, w_img, h_img)
                    if mar > MAR_THRESHOLD: is_yawning = True
                    
                    def get_eye_roi(landmark_idxs, padding=10):
                        coords = [(int(face_landmarks.landmark[i].x * w_img), int(face_landmarks.landmark[i].y * h_img)) for i in landmark_idxs]
                        x_min = max(0, min([c[0] for c in coords]) - padding)
                        x_max = min(w_img, max([c[0] for c in coords]) + padding)
                        y_min = max(0, min([c[1] for c in coords]) - padding)
                        y_max = min(h_img, max([c[1] for c in coords]) + padding)
                        return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

                    left_eye_img, l_rect = get_eye_roi(LEFT_EYE_IDXS)
                    right_eye_img, r_rect = get_eye_roi(RIGHT_EYE_IDXS)
                    
                    score_l = predict_eye_status(left_eye_img)
                    score_r = predict_eye_status(right_eye_img)
                    
                    cv2.rectangle(frame, (l_rect[0], l_rect[1]), (l_rect[2], l_rect[3]), (0, 255, 255), 1)
                    cv2.rectangle(frame, (r_rect[0], r_rect[1]), (r_rect[2], r_rect[3]), (0, 255, 255), 1)

                    if score_l <= EYE_CONFIDENCE_THRESHOLD or score_r <= EYE_CONFIDENCE_THRESHOLD: status_text = "Closed"
                    else: status_text = "Open"
                    
                    pts2d = []
                    for idx in landmark_ids_pose:
                        lm = face_landmarks.landmark[idx]
                        pts2d.append([lm.x * w_img, lm.y * h_img])
                    pts2d = np.array(pts2d, dtype=np.float64)
                    
                    focal_length = w_img
                    center = (w_img/2, h_img/2)
                    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
                    dist_coeffs = np.zeros((4,1))
                    success, rot_vec, trans_vec = cv2.solvePnP(model_points, pts2d, camera_matrix, dist_coeffs)
                    if success:
                        rmat, _ = cv2.Rodrigues(rot_vec)
                        proj = np.hstack((rmat, trans_vec))
                        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
                        pitch, yaw, roll = euler.flatten()

            if not face_detected:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60,60))
                
                if len(faces) > 0:
                    face_detected = True
                    status_text = "Open"
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 1)
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = frame[y:y+h, x:x+w]
                        eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20,20))
                        if len(eyes) > 0:
                            for (ex, ey, ew, eh) in eyes:
                                eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                                cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 255), 1)
                                score = predict_eye_status(eye_roi)
                                if score <= EYE_CONFIDENCE_THRESHOLD: status_text = "Closed"

            if face_detected:
                is_distracted = (abs(yaw) > YAW_DISTRACTION_THRESHOLD) or (abs(pitch) < PITCH_THRESHOLD) or (abs(roll) > ROLL_THRESHOLD)
                is_closed = (status_text == "Closed")
            else:
                is_closed = False
                is_distracted = False
            
            raw_cam_fatigue = calculate_raw_camera_fatigue(face_detected, is_closed, is_yawning, is_distracted)
            
            current_alcohol = sensor_data.get("alcohol_ppm", 0.0)
            current_smoke   = sensor_data.get("smoke_ppm", 0.0)
            current_hr      = sensor_data.get("heart_rate", 0)
            current_body_t  = sensor_data.get("temperature_2", 0.0)
            current_cabin_t = sensor_data.get("temperature_1", 0.0)
            
            _, fuzzy_texts = FuzzyLogicEngine.evaluate_all(
                cam_fatigue=raw_cam_fatigue, 
                alc_ppm=current_alcohol,
                smoke_ppm=current_smoke,
                heart_rate=current_hr,
                body_temp=current_body_t,
                cabin_temp=current_cabin_t
            )
            
            sensor_data["fatigue_score"] = raw_cam_fatigue 
            sensor_data["fuzzy_alcohol"] = fuzzy_texts["alcohol"]
            sensor_data["fuzzy_smoke"] = fuzzy_texts["smoke"]
            sensor_data["fuzzy_heart_rate"] = fuzzy_texts["heart_rate"]
            sensor_data["fuzzy_body_temp"] = fuzzy_texts["body_temp"]
            sensor_data["fuzzy_cabin_temp"] = fuzzy_texts["cabin_temp"]
            
            # --- STATUS OVERRIDE ---
            if sensor_data.get("crash_detected", 0) == 1:
                sensor_data["status"] = "CRASH DETECTED"
            elif raw_cam_fatigue > 80:
                sensor_data["status"] = "CRITICAL RISK"
            elif "Danger" in fuzzy_texts["alcohol"]:
                sensor_data["status"] = "ALCOHOL DANGER"
            elif "Danger" in fuzzy_texts["smoke"]:
                sensor_data["status"] = "SMOKE DANGER"
            elif raw_cam_fatigue > 50:
                sensor_data["status"] = "WARNING"
            elif sensor_data["status"] != "CALIBRATING":
                sensor_data["status"] = "SAFE"

            sensor_data["last_pitch"] = pitch
            sensor_data["last_roll"] = roll
            sensor_data["last_yaw"] = yaw
            sensor_data["last_eyes"] = status_text

            draw_hud(frame, pitch, roll, yaw, status_text, raw_cam_fatigue, is_yawning)

            if best_face:
                cx, cy = int(best_face.landmark[1].x * w_img), int(best_face.landmark[1].y * h_img)
                cv2.rectangle(frame, (cx-90, cy-90), (cx+90, cy+90), (0, 255, 0), 2)
                cv2.putText(frame, "DRIVER LOCKED", (cx-90, cy-100), font, 0.5, (0, 255, 0), 1)

            if not face_detected:
                cv2.putText(frame, "NO FACE DETECTED", (180, 240), font, 0.8, (200, 200, 200), 2)
            
            with frame_lock: output_frame = frame.copy()
            time.sleep(0.01)

        except Exception as e:
            print("Loop Error:", e)
            time.sleep(0.1)

def generate_mjpeg():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None: 
                time.sleep(0.1)
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            if not flag: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.06)

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/sensors', methods=['GET'])
def get_sensors():
    return jsonify(sensor_data)

if __name__ == '__main__':
    gps_thread = threading.Thread(target=run_gps_system)
    gps_thread.daemon = True
    gps_thread.start()
    
    arduino_thread = threading.Thread(target=run_arduino_bridge)
    arduino_thread.daemon = True
    arduino_thread.start()
    
    buzzer_thread = threading.Thread(target=run_buzzer_control)
    buzzer_thread.daemon = True
    buzzer_thread.start()
    
    logger_thread = threading.Thread(target=run_data_logger)
    logger_thread.daemon = True
    logger_thread.start()

    t = threading.Thread(target=run_detection_system)
    t.daemon = True
    t.start()
    
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)