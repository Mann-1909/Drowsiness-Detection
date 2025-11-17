#!/usr/bin/env python3
import time
import os
import cv2
import numpy as np

# Use Picamera2
from picamera2 import Picamera2

# Try to use tflite runtime first (recommended on Pi). If not available fall back to TF Keras.
TFLITE_AVAILABLE = False
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    TFLITE_AVAILABLE = True
except Exception:
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
    except Exception:
        tf = None

# Try import mediapipe (optional)
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    USE_MEDIAPIPE = True
except Exception:
    USE_MEDIAPIPE = False
    print("mediapipe not available — head-pose estimation will be skipped.")

# -------------------------
# Config - change these as needed
# -------------------------
CASCADE_PATH = "/home/pi/Desktop/DDD/haarcascade/"   # your cascade folder
TFLITE_MODEL_PATH = "/home/pi/Desktop/DDD/my_model.tflite"  # change if you have .tflite
KERAS_MODEL_PATH  = "/home/pi/Desktop/DDD/my_model.keras"     # fallback Keras model path

# thresholds (tweak to taste)
EYE_CONFIDENCE_THRESHOLD = 0.5    # model score threshold (<= => closed)
ALERT_THRESHOLD_SEC = 2.0
PITCH_THRESHOLD = 160.0
ROLL_THRESHOLD = 38.0
PITCH_VELOCITY_THRESHOLD = 30.0  # degrees per second

# -------------------------
# Load cascades (local)
# -------------------------
faceCascade = cv2.CascadeClassifier(os.path.join(CASCADE_PATH, "haarcascade_frontalface_default.xml"))
eyeCascade  = cv2.CascadeClassifier(os.path.join(CASCADE_PATH, "haarcascade_eye.xml"))

if faceCascade.empty():
    raise IOError("Cannot load face cascade from: " + os.path.join(CASCADE_PATH, "haarcascade_frontalface_default.xml"))
if eyeCascade.empty():
    raise IOError("Cannot load eye cascade from: " + os.path.join(CASCADE_PATH, "haarcascade_eye.xml"))

# -------------------------
# Load model (TFLite preferred)
# -------------------------
using_tflite = False
interpreter = None
input_details = output_details = None
keras_model = None
in_height = in_width = in_channels = None

if os.path.exists(TFLITE_MODEL_PATH) and TFLITE_AVAILABLE:
    try:
        interpreter = TFLiteInterpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # infer input size
        in_height = int(input_details[0]['shape'][1])
        in_width  = int(input_details[0]['shape'][2])
        in_channels = int(input_details[0]['shape'][3])
        using_tflite = True
        print("Using TFLite model:", TFLITE_MODEL_PATH)
        print("TFLite input shape:", input_details[0]['shape'])
    except Exception as e:
        print("Failed to load TFLite model:", e)
        interpreter = None

if not using_tflite:
    # fallback to Keras if available
    if os.path.exists(KERAS_MODEL_PATH):
        try:
            if tf is None:
                raise RuntimeError("TensorFlow not available to load Keras model.")
            keras_model = load_model(KERAS_MODEL_PATH)
            # infer input size if possible
            try:
                shape = keras_model.input_shape
                # shape may be (None, H, W, C)
                in_height = shape[1]
                in_width  = shape[2]
                in_channels = shape[3]
            except Exception:
                in_height = in_width = 224
                in_channels = 3
            print("Using Keras model:", KERAS_MODEL_PATH)
        except Exception as e:
            raise RuntimeError("Failed to load Keras model: " + str(e))
    else:
        raise RuntimeError("No TFLite model found and Keras model not found. Put either a .tflite or .keras model in paths.")

# -------------------------
# Mediapipe setup if available
# -------------------------
if USE_MEDIAPIPE:
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
    # landmark ids and 3D model points for solvePnP (same as your code)
    landmark_ids = [1, 152, 33, 263, 61, 291]
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype=np.float64)

# -------------------------
# Picamera2 setup
# -------------------------
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "YUV420"}
)
picam2.configure(config)
picam2.start()

time.sleep(0.1)
print("Camera started. Press 'q' to quit.")

# -------------------------
# State variables
# -------------------------
prev_pitch = None
prev_time = None
alert_start_time = None

# -------------------------
# Main loop
# -------------------------
try:
    while True:
        yuv = picam2.capture_array()
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        status = "No Face"

        # face detection
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60,60))
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # eye detection inside face
            eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20,20))
            if len(eyes) == 0:
                status = "Closed Eyes"
            else:
                any_closed = False
                # for each detected eye run model (if ROI valid)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,0,255), 2)
                    eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                    if eye_roi.size == 0:
                        continue

                    # preprocess for the model
                    h_in = in_height or 224
                    w_in = in_width or 224
                    resized = cv2.resize(eye_roi, (w_in, h_in))
                    input_tensor = resized.astype(np.float32) / 255.0
                    input_tensor = np.expand_dims(input_tensor, axis=0)

                    # inference
                    try:
                        if using_tflite and interpreter is not None:
                            interpreter.set_tensor(input_details[0]['index'], input_tensor)
                            interpreter.invoke()
                            pred = interpreter.get_tensor(output_details[0]['index'])
                        else:
                            pred = keras_model.predict(input_tensor)
                    except Exception as e:
                        print("Inference error:", e)
                        pred = np.array([[1.0]])  # assume open to avoid false positives

                    score = float(np.array(pred).flatten()[0])
                    if score <= EYE_CONFIDENCE_THRESHOLD:
                        any_closed = True

                status = "Closed Eyes" if any_closed else "Open Eyes"

        # head-pose using mediapipe (optional)
        pitch = yaw = roll = 0.0
        if USE_MEDIAPIPE:
            rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_small)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h_img, w_img = frame.shape[:2]
                pts2d = []
                for idx in landmark_ids:
                    lm = face_landmarks.landmark[idx]
                    pts2d.append([lm.x * w_img, lm.y * h_img])
                pts2d = np.array(pts2d, dtype=np.float64)

                focal_length = w_img
                center = (w_img/2, h_img/2)
                camera_matrix = np.array([[focal_length, 0, center[0]],
                                          [0, focal_length, center[1]],
                                          [0, 0, 1]], dtype=np.float64)
                dist_coeffs = np.zeros((4,1))

                success, rot_vec, trans_vec = cv2.solvePnP(model_points, pts2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                if success:
                    rmat, _ = cv2.Rodrigues(rot_vec)
                    proj = np.hstack((rmat, trans_vec))
                    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
                    pitch, yaw, roll = euler.flatten()

                    # pitch velocity (optional)
                    curr_time = time.time()
                    if prev_pitch is not None and prev_time is not None:
                        dt = curr_time - prev_time
                        if dt > 0:
                            pitch_velocity = (pitch - prev_pitch) / dt
                    prev_pitch = pitch
                    prev_time = curr_time

                    cv2.putText(frame, f"Pitch:{int(pitch)} Yaw:{int(yaw)} Roll:{int(roll)}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                    cv2.putText(frame, f"Eye:{status}", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

                    # drowsiness rule example
                    if (abs(pitch) < PITCH_THRESHOLD) or (abs(roll) > ROLL_THRESHOLD) or (status == "Closed Eyes"):
                        if alert_start_time is None:
                            alert_start_time = time.time()
                        elif time.time() - alert_start_time > ALERT_THRESHOLD_SEC:
                            cv2.putText(frame, "DROWSINESS ALERT!", (int(w_img*0.25), int(h_img*0.4)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                    else:
                        alert_start_time = None
        else:
            # simple eye-based alert if mediapipe unavailable
            if status == "Closed Eyes":
                if alert_start_time is None:
                    alert_start_time = time.time()
                elif time.time() - alert_start_time > ALERT_THRESHOLD_SEC:
                    cv2.putText(frame, "DROWSINESS ALERT!", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            else:
                alert_start_time = None

        # display
        cv2.imshow("Drowsiness Detection (Picamera2)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    if USE_MEDIAPIPE:
        face_mesh.close()
