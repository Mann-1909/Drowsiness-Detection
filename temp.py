#!/usr/bin/env python3
import time
import os
import cv2
import numpy as np
import RPi.GPIO as GPIO
import spidev
import threading
import math
import serial
from flask import Flask, jsonify, Response
from picamera2 import Picamera2

# --- FLASK APP SETUP ---
app = Flask(__name__)

# --- GLOBAL VARIABLES ---
output_frame = None
frame_lock = threading.Lock()

# Shared Data Dictionary
sensor_data = {
    "alcohol_level": 0,
    "heart_rate": 0,
    "status": "SAFE",
    "latitude": 0.0,
    "longitude": 0.0
}

# --- HARDWARE SETUP ---
GPIO.setmode(GPIO.BCM)
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000 

buzzer_pin = 17
GPIO.setup(buzzer_pin, GPIO.OUT)
GPIO.output(buzzer_pin, GPIO.LOW)

# --- CONFIGURATION ---
CASCADE_PATH = "/home/pi/Desktop/DDD/haarcascade/"
TFLITE_MODEL_PATH = "/home/pi/Desktop/DDD/my_model.tflite"

# Thresholds
EYE_CONFIDENCE_THRESHOLD = 0.5
ALERT_THRESHOLD_SEC = 2.0
PITCH_THRESHOLD = 160
ROLL_THRESHOLD = 38

# New Sensor Thresholds
ALCOHOL_THRESHOLD = 400   # Adjust based on your sensor calibration
BPM_HIGH_THRESHOLD = 120  # Alert if heart rate is too high
BPM_LOW_THRESHOLD = 40    # Alert if heart rate is too low

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

# --- ROBUST MEDIAPIPE IMPORT ---
USE_MEDIAPIPE = False
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    USE_MEDIAPIPE = True
except Exception as e:
    USE_MEDIAPIPE = False
    print(f"MediaPipe Error: {e}. System will rely on Haar Cascades.")

# --- ANALOG READ ---
def analog_read(channel):
    if channel < 0 or channel > 7: return -1
    r = spi.xfer2([1, (8 + channel) << 4, 0])
    return ((r[1] & 3) << 8) + r[2]

# --- NEW FUNCTION: CALCULATE STATUS ---
def calculate_system_status(face_detected, pitch, roll, eye_text, alcohol_val, bpm, alert_start_time):
    """
    Determines the system status based on all sensors.
    """
    # 1. Check Alcohol
    if alcohol_val > ALCOHOL_THRESHOLD:
        return "ALCOHOL DETECTED", None

    # 2. Check Heart Rate
    if bpm > 0 and (bpm > BPM_HIGH_THRESHOLD or bpm < BPM_LOW_THRESHOLD):
        return "VITAL ALERT", None

    # 3. Check Drowsiness
    if face_detected:
        if (abs(pitch) < PITCH_THRESHOLD) or (abs(roll) > ROLL_THRESHOLD) or (eye_text == "Closed"):
            if alert_start_time is None:
                return "SAFE", time.time()
            elif time.time() - alert_start_time > ALERT_THRESHOLD_SEC:
                return "DROWSINESS DETECTED", alert_start_time
            else:
                return "SAFE", alert_start_time
        else:
            return "SAFE", None
    else:
        return "SAFE", None

# --- BUZZER CONTROL THREAD ---
def run_buzzer_control():
    global sensor_data
    print("--- Buzzer System Started ---")
    while True:
        status = sensor_data["status"]
        if status == "ALCOHOL DETECTED":
            GPIO.output(buzzer_pin, GPIO.HIGH)
            time.sleep(1.0)
            GPIO.output(buzzer_pin, GPIO.LOW)
            time.sleep(0.2)
        elif status == "VITAL ALERT":
            GPIO.output(buzzer_pin, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(buzzer_pin, GPIO.LOW)
            time.sleep(0.1)
        elif status == "DROWSINESS DETECTED":
            GPIO.output(buzzer_pin, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(buzzer_pin, GPIO.LOW)
            time.sleep(0.5)
        else:
            GPIO.output(buzzer_pin, GPIO.LOW)
            time.sleep(0.2)

# --- CLASS: HEART RATE MONITOR ---
class HeartRateMonitor:
    def __init__(self):
        self.bpm = 0
        self.last_beat_time = time.time()
        self.threshold = 575  
        self.finger_threshold = 100 
        self.ibi_list = []    
        self.min_diff = 0.3   
        self.beat_detected = False

    def process_sample(self, signal_value):
        current_time = time.time()
        if signal_value < self.finger_threshold:
            self.bpm = 0
            self.ibi_list = []
            self.beat_detected = False
            self.last_beat_time = current_time 
            return 0
        if (current_time - self.last_beat_time) > 2.5:
            self.bpm = 0
            self.ibi_list = []
        if signal_value > self.threshold and not self.beat_detected:
            time_diff = current_time - self.last_beat_time
            if time_diff > self.min_diff:
                self.beat_detected = True
                self.last_beat_time = current_time
                instant_bpm = 60.0 / time_diff
                if 40 < instant_bpm < 180: 
                    self.ibi_list.append(instant_bpm)
                    if len(self.ibi_list) > 10:
                        self.ibi_list.pop(0)
                    self.bpm = int(sum(self.ibi_list) / len(self.ibi_list))
        if signal_value < self.threshold:
            self.beat_detected = False
        return self.bpm

# --- THREADS ---
def run_heart_monitor():
    global sensor_data
    monitor = HeartRateMonitor()
    print("--- Heart Rate System Started ---")
    while True:
        raw_val = analog_read(1)
        bpm = monitor.process_sample(raw_val)
        sensor_data["heart_rate"] = bpm
        time.sleep(0.01)

def convert_to_degrees(raw_value):
    try:
        decimal_point_position = raw_value.find('.')
        if decimal_point_position == -1: return 0.0
        degrees_digits = decimal_point_position - 2
        degrees = float(raw_value[:degrees_digits])
        minutes = float(raw_value[degrees_digits:])
        return degrees + (minutes / 60)
    except Exception:
        return 0.0

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
                        lat = convert_to_degrees(parts[2])
                        lon = convert_to_degrees(parts[4])
                        if parts[3] == 'S': lat = -lat
                        if parts[5] == 'W': lon = -lon
                        sensor_data["latitude"] = lat
                        sensor_data["longitude"] = lon
            except Exception:
                pass
    except Exception as e:
        print(f"GPS Init Failed: {e}")

# --- HELPER: PREDICT EYE STATUS ---
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
    except Exception as e:
        pass
    return 1.0

# --- HELPER: DRAW HUD (REVERTED VISUALS) ---
def draw_hud(frame, pitch, roll, yaw, eye_status, is_drowsy, bpm):
    h, w, _ = frame.shape
    overlay = frame.copy()
    
    # Original Top Bar Height (60)
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    status_color = (0, 255, 0) if not is_drowsy else (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Original Text Positions
    cv2.putText(frame, f"Pitch: {int(pitch)}", (20, 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Roll: {int(roll)}", (140, 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Yaw: {int(yaw)}", (260, 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Eyes: {eye_status}", (20, 50), font, 0.6, status_color, 2, cv2.LINE_AA)
    
    # Added BPM to top row (Space permitting)
    cv2.putText(frame, f"BPM: {bpm}", (380, 25), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

# --- MAIN DETECTION LOOP ---
def run_detection_system():
    global output_frame, sensor_data
    
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480), "format": "YUV420"})
    picam2.configure(config)
    picam2.start()
    
    # Only FaceMesh (No Pose/Skeleton)
    if USE_MEDIAPIPE:
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
    
    # Landmark definitions
    LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
    landmark_ids_pose = [1, 152, 33, 263, 61, 291]
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ], dtype=np.float64)

    alert_start_time = None
    print("--- Detection System Running ---")

    while True:
        try:
            # 1. READ SENSORS
            sensor_data["alcohol_level"] = analog_read(0)
            current_bpm = sensor_data["heart_rate"]
            
            # 2. CAPTURE IMAGE
            yuv = picam2.capture_array()
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            h_img, w_img = frame.shape[:2]
            
            status_text = "No Face"
            pitch = roll = yaw = 0.0
            face_detected = False
            
            # 3. MEDIAPIPE PROCESSING
            if USE_MEDIAPIPE:
                rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_small)
                
                # --- FACE LOGIC ---
                if results.multi_face_landmarks:
                    face_detected = True
                    face_landmarks = results.multi_face_landmarks[0]
                    
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

                    if score_l <= EYE_CONFIDENCE_THRESHOLD or score_r <= EYE_CONFIDENCE_THRESHOLD:
                        status_text = "Closed"
                    else:
                        status_text = "Open"
                    
                    # Pose Logic (Head Rotation)
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
                        # Draw Axis (Nose line)
                        axis_points = np.float32([[100,0,0], [0,100,0], [0,0,100]])
                        imgpts, _ = cv2.projectPoints(axis_points, rot_vec, trans_vec, camera_matrix, dist_coeffs)
                        
                        # Helper for axis drawing
                        nose_tip = tuple(pts2d[0].astype(int))
                        pt_x = tuple(imgpts[0].ravel().astype(int))
                        pt_y = tuple(imgpts[1].ravel().astype(int))
                        pt_z = tuple(imgpts[2].ravel().astype(int))
                        cv2.line(frame, nose_tip, pt_x, (0, 0, 255), 3)
                        cv2.line(frame, nose_tip, pt_y, (0, 255, 0), 3)
                        cv2.line(frame, nose_tip, pt_z, (255, 0, 0), 3)

                        rmat, _ = cv2.Rodrigues(rot_vec)
                        proj = np.hstack((rmat, trans_vec))
                        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
                        pitch, yaw, roll = euler.flatten()

            # 4. FALLBACK TO HAAR (BACKUP SYSTEM)
            if not face_detected:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60,60))
                
                if len(faces) > 0:
                    face_detected = True
                    status_text = "Open" 
                    for (x, y, w, h) in faces:
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = frame[y:y+h, x:x+w]
                        eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20,20))
                        
                        if len(eyes) > 0:
                            for (ex, ey, ew, eh) in eyes:
                                eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                                cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 255), 1)
                                score = predict_eye_status(eye_roi)
                                if score <= EYE_CONFIDENCE_THRESHOLD:
                                    status_text = "Closed"
                                else:
                                    status_text = "Open"

            # 5. CALCULATE STATUS
            calculated_status, alert_start_time = calculate_system_status(
                face_detected, pitch, roll, status_text, 
                sensor_data["alcohol_level"], current_bpm, alert_start_time
            )

            sensor_data["status"] = calculated_status

            # 6. OUTPUT & ALERTS (VISUALS)
            is_drowsy_visual = calculated_status != "SAFE"
            
            # Draw Standard HUD
            draw_hud(frame, pitch, roll, yaw, status_text, is_drowsy_visual, current_bpm)

            # Draw BIG Center Alerts (Like Previous Code)
            if not face_detected:
                cv2.putText(frame, "NO FACE DETECTED", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            if calculated_status == "DROWSINESS DETECTED":
                cv2.putText(frame, "DROWSINESS ALERT!", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            elif calculated_status == "ALCOHOL DETECTED":
                cv2.putText(frame, "ALCOHOL ALERT!", (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            elif calculated_status == "VITAL ALERT":
                cv2.putText(frame, "HEART RATE ALERT!", (160, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

            with frame_lock:
                output_frame = frame.copy()
            
            time.sleep(0.01)

        except Exception as e:
            print("Loop Error:", e)

# --- FLASK STREAMING ---
def generate_mjpeg():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None: continue
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
    
    hr_thread = threading.Thread(target=run_heart_monitor)
    hr_thread.daemon = True
    hr_thread.start()
    
    buzzer_thread = threading.Thread(target=run_buzzer_control)
    buzzer_thread.daemon = True
    buzzer_thread.start()

    t = threading.Thread(target=run_detection_system)
    t.daemon = True
    t.start()
    
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)