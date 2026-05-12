#!/usr/bin/env python3
import time
import os
import cv2
import numpy as np
import RPi.GPIO as GPIO
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

# Shared Data Dictionary
sensor_data = {
    "alcohol_ppm": 0.0,
    "smoke_ppm": 0.0,
    "heart_rate": 0,
    "temperature_1": 0.0,
    "temperature_2": 0.0,
    "cpu_temp": 0.0,
    "status": "BOOTING",
    "fatigue_score": 0.0,
    "latitude": 0.0,
    "longitude": 0.0,
    "is_night_mode": False
}

# --- HARDWARE SETUP ---
GPIO.setmode(GPIO.BCM)
buzzer_pin = 17
GPIO.setup(buzzer_pin, GPIO.OUT)
GPIO.output(buzzer_pin, GPIO.LOW)

# Create Evidence Folder
EVIDENCE_PATH = "/home/pi/Desktop/DDD/evidence"
if not os.path.exists(EVIDENCE_PATH): os.makedirs(EVIDENCE_PATH)

# --- CONFIGURATION ---
CASCADE_PATH = "/home/pi/Desktop/DDD/haarcascade/"
TFLITE_MODEL_PATH = "/home/pi/Desktop/DDD/my_model.tflite"

# Performance Settings
ENABLE_SKELETON = True
FRAME_SKIP = 2

# Safety Thresholds (Updated for PPM/Celsius)
ALCOHOL_LIMIT_PPM = 100  # PPM limit for Alcohol
SMOKE_LIMIT_PPM = 400.0   # PPM limit for Smoke/LPG
BPM_HIGH_THRESHOLD = 120  
BPM_LOW_THRESHOLD = 40    
TEMP_HIGH_THRESHOLD = 40.0 # Celsius
CPU_WARN_THRESHOLD = 75.0

# Vision Thresholds
EYE_CONFIDENCE_THRESHOLD = 0.5
MAR_THRESHOLD = 0.5        
PITCH_THRESHOLD = 25       
ROLL_THRESHOLD = 25        
YAW_DISTRACTION_THRESHOLD = 30 

# Weights for Risk Calculation
WEIGHT_CAMERA = 0.6        
WEIGHT_ALCOHOL = 1.0       
WEIGHT_VITALS = 0.4        

SCORE_DECAY = 0.5          
SCORE_EYES_CLOSED = 2.0    
SCORE_YAWN = 1.0           
SCORE_DISTRACTED = 1.5     

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
    with open("/home/pi/Desktop/DDD/error_log.txt", "w") as f:
        f.write(f"MediaPipe Import Error: {str(e)}")
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
    print(f"Evidence Saved: {filename}")

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

# --- SENSOR FUSION ENGINE ---
def normalize_sensor_risk(val, min_safe, max_danger):
    if val < min_safe: return 0.0
    if val > max_danger: return 1.0
    return (val - min_safe) / (max_danger - min_safe)

def normalize_bpm_risk(bpm):
    if bpm == 0: return 0.0 
    if bpm < BPM_LOW_THRESHOLD: return 0.8 
    if bpm > BPM_HIGH_THRESHOLD: return 0.8 
    return 0.0

current_camera_score = 0.0

def calculate_total_risk(face_detected,is_closed, is_yawning, is_distracted, alc_ppm, bpm, temp):
    global current_camera_score
    if not face_detected: current_camera_score = 0.0
    elif is_closed: current_camera_score += SCORE_EYES_CLOSED
    elif is_yawning: current_camera_score += SCORE_YAWN
    elif is_distracted: current_camera_score += SCORE_DISTRACTED
    else: current_camera_score -= SCORE_DECAY
    current_camera_score = max(0.0, min(100.0, current_camera_score))
    
    # Updated to use PPM limits
    risk_alcohol = normalize_sensor_risk(alc_ppm, 0.1, ALCOHOL_LIMIT_PPM)
    risk_bpm = normalize_bpm_risk(bpm)
    risk_temp = normalize_sensor_risk(temp, 37, TEMP_HIGH_THRESHOLD)
    
    base_risk = current_camera_score / 100.0
    total_risk_val = (base_risk * WEIGHT_CAMERA) + (risk_alcohol * WEIGHT_ALCOHOL) + (max(risk_bpm, risk_temp) * WEIGHT_VITALS)
    return min(100.0, total_risk_val * 100.0)

# --- THREADS ---

# NEW: ARDUINO SERIAL BRIDGE (UPDATED FOR 5 VALUES)
def run_arduino_bridge():
    global sensor_data
    print("--- Scanning for Arduino... ---")
    arduino = None
    
    ports = ['/dev/ttyACM0', '/dev/ttyUSB0', '/dev/ttyACM1', '/dev/ttyUSB1']
    for p in ports:
        try:
            arduino = serial.Serial(p, 9600, timeout=1)
            print(f"SUCCESS: Arduino Connected on {p}")
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
                
                # Expecting: "ALC_PPM, SMOKE_PPM, BPM, TEMP1, TEMP2"
                parts = line.split(',')
                if len(parts) >= 5:
                    alc_ppm = float(parts[0])
                    smoke_ppm = float(parts[1])
                    bpm = int(parts[2])
                    temp1 = float(parts[3])
                    temp2 = float(parts[4])
                    
                    sensor_data["alcohol_ppm"] = alc_ppm
                    sensor_data["smoke_ppm"] = smoke_ppm
                    sensor_data["heart_rate"] = bpm
                    sensor_data["temperature_1"] = temp1
                    sensor_data["temperature_2"] = temp2
                    
                    # Override status for high alcohol or smoke
                    if alc_ppm > ALCOHOL_LIMIT_PPM: 
                        sensor_data["status"] = "ALCOHOL DETECTED"
                    elif smoke_ppm > SMOKE_LIMIT_PPM:
                        sensor_data["status"] = "SMOKE DETECTED"
                        
        except Exception as e:
            # print(f"Serial Error: {e}") 
            time.sleep(0.5)

def run_data_logger():
    filename = datetime.now().strftime("/home/pi/Desktop/DDD/trip_log_%Y%m%d_%H%M%S.csv")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Status", "FatigueScore", "BPM", "Alcohol_PPM", "Smoke_PPM", "Temp1", "Temp2", "CPUTemp", "Lat", "Lon"])
    print(f"--- Data Logger Started: {filename} ---")
    while True:
        try:
            sensor_data["cpu_temp"] = get_cpu_temperature()
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                timestamp = datetime.now().strftime("%H:%M:%S")
                writer.writerow([timestamp, sensor_data["status"], round(sensor_data["fatigue_score"], 1),
                                sensor_data["heart_rate"], sensor_data["alcohol_ppm"], sensor_data["smoke_ppm"], 
                                sensor_data["temperature_1"], sensor_data["temperature_2"], sensor_data["cpu_temp"],
                                sensor_data["latitude"], sensor_data["longitude"]])
            time.sleep(1.0)
        except: time.sleep(1.0)

def run_buzzer_control():
    global sensor_data
    print("--- Buzzer/Voice System Started ---")
    last_spoken_time = 0
    last_snapshot_time = 0
    while True:
        score = sensor_data["fatigue_score"]
        status = sensor_data["status"]
        cpu = sensor_data["cpu_temp"]
        current_time = time.time()

        if cpu > CPU_WARN_THRESHOLD:
            if current_time - last_spoken_time > 15:
                speak_warning("Warning. System Overheating.")
                last_spoken_time = current_time

        if status == "CALIBRATING":
            time.sleep(1)
            continue
        
        if status == "ALCOHOL DETECTED":
            GPIO.output(buzzer_pin, GPIO.HIGH)
            if current_time - last_spoken_time > 5:
                speak_warning("Alcohol Detected. Pull over.")
                last_spoken_time = current_time
            time.sleep(1.0)
            GPIO.output(buzzer_pin, GPIO.LOW)
            time.sleep(0.2)
        elif status == "SMOKE DETECTED":
            GPIO.output(buzzer_pin, GPIO.HIGH)
            if current_time - last_spoken_time > 5:
                speak_warning("Fire Hazard Detected.")
                last_spoken_time = current_time
            time.sleep(0.5)
            GPIO.output(buzzer_pin, GPIO.LOW)
            time.sleep(0.1)
        elif score > 80:
            GPIO.output(buzzer_pin, GPIO.HIGH)
            if current_time - last_spoken_time > 4:
                speak_warning("Critical Alert. Wake up.")
                last_spoken_time = current_time
            if current_time - last_snapshot_time > 5:
                with frame_lock:
                    if output_frame is not None: save_evidence_snapshot(output_frame, "FATIGUE")
                last_snapshot_time = current_time
            time.sleep(0.5)
            GPIO.output(buzzer_pin, GPIO.LOW)
            time.sleep(0.1)
        elif score > 50:
            GPIO.output(buzzer_pin, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(buzzer_pin, GPIO.LOW)
            if current_time - last_spoken_time > 10:
                speak_warning("You are drowsy.")
                last_spoken_time = current_time
            time.sleep(2.0)
        else:
            GPIO.output(buzzer_pin, GPIO.LOW)
            time.sleep(0.1)

# GPS Thread
def run_gps_system():
    global sensor_data
    print("--- GPS System Started ---")
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
                        # Conversion logic
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
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    
    if calibration_progress is not None:
        bar_w = int((calibration_progress / 100.0) * w)
        cv2.rectangle(overlay, (0, h//2 - 20), (bar_w, h//2 + 20), (0, 255, 255), -1)
        cv2.putText(frame, "CALIBRATING...", (w//2 - 100, h//2 + 10), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return

    bar_width = int((score / 100.0) * w)
    bar_color = (0, 255, 0)
    if score > 50: bar_color = (0, 255, 255)
    if score > 80: bar_color = (0, 0, 255)
    cv2.rectangle(overlay, (0, h - 20), (bar_width, h), bar_color, -1)

    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, f"Pitch: {int(pitch)}", (20, 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Roll: {int(roll)}", (140, 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Yaw: {int(yaw)}", (260, 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    eye_color = (0, 0, 255) if eye_status == "Closed" else (0, 255, 0)
    cv2.putText(frame, f"Eyes: {eye_status}", (20, 55), font, 0.6, eye_color, 2, cv2.LINE_AA)
    if yawn_status: cv2.putText(frame, "YAWN!", (160, 55), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Risk: {int(score)}%", (w - 150, 55), font, 0.6, bar_color, 2, cv2.LINE_AA)
    if sensor_data["is_night_mode"]: cv2.putText(frame, "NIGHT VISION", (w - 150, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    if sensor_data["cpu_temp"] > 70: cv2.putText(frame, f"CPU: {int(sensor_data['cpu_temp'])}C", (w - 300, 25), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    if score > 80: cv2.putText(frame, "CRITICAL RISK!", (w//2 - 150, h//2), font, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    elif score > 50: cv2.putText(frame, "WARNING: Fatigue", (w//2 - 120, h//2), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

# --- MAIN DETECTION LOOP ---
def run_detection_system():
    global output_frame, sensor_data
    
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480), "format": "YUV420"})
    picam2.configure(config)
    picam2.start()
    
    if USE_MEDIAPIPE:
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
    landmark_ids_pose = [1, 152, 33, 263, 61, 291]
    model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)], dtype=np.float64)

    # Calibration Phase
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
            # 1. Capture
            yuv = picam2.capture_array()
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            frame = adjust_gamma(frame, gamma=1.2)
            h_img, w_img = frame.shape[:2]
            
            # --- START FRAME SKIP ---
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                draw_hud(frame, sensor_data.get("last_pitch", 0), sensor_data.get("last_roll", 0), sensor_data.get("last_yaw", 0), 
                         sensor_data.get("last_eyes", "Open"), sensor_data["fatigue_score"], False)
                with frame_lock: output_frame = frame.copy()
                time.sleep(0.01)
                continue
            # --- END FRAME SKIP ---

            status_text = "No Face"
            pitch = roll = yaw = 0.0
            face_detected = False
            is_yawning = False
            
            if USE_MEDIAPIPE:
                rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # --- SKELETON ---
                if ENABLE_SKELETON:
                    pose_results = pose.process(rgb_small)
                    if pose_results.pose_landmarks:
                        landmarks = pose_results.pose_landmarks.landmark
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

                # --- FACE MESH ---
                results = face_mesh.process(rgb_small)
                if results.multi_face_landmarks:
                    face_detected = True
                    face_landmarks = results.multi_face_landmarks[0]
                    
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
                # --- RESTORED HAAR CASCADE FALLBACK ---
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
            
            final_risk = calculate_total_risk(face_detected,
                is_closed, is_yawning, is_distracted, 
                sensor_data["alcohol_ppm"], 
                sensor_data["heart_rate"], 
                sensor_data["temperature_1"] # Use Temp1 as cabin temp
            )
            sensor_data["fatigue_score"] = final_risk
            
            sensor_data["last_pitch"] = pitch
            sensor_data["last_roll"] = roll
            sensor_data["last_yaw"] = yaw
            sensor_data["last_eyes"] = status_text

            draw_hud(frame, pitch, roll, yaw, status_text, final_risk, is_yawning)

            if not face_detected:
                cv2.putText(frame, "NO FACE DETECTED", (180, 240), font, 0.8, (200, 200, 200), 2)
            
            with frame_lock: output_frame = frame.copy()
            time.sleep(0.01)

        except Exception as e:
            print("Loop Error:", e)
            time.sleep(0.1)

# --- FLASK STREAMING ---
def generate_mjpeg():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None: 
                time.sleep(0.1)
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

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
    
    # ARDUINO SERIAL THREAD
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