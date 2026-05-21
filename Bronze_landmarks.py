import cv2
import mediapipe as mp
import json
import time
import threading
from confluent_kafka import Producer
import boto3
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

STUDENT_ID = "Szymon Skarbek"

KAFKA_SERVER = 'localhost:8081'
MINIO_URL = 'http://localhost:9005'

MINIO_ACCESS_KEY = 'admin'
MINIO_SECRET_KEY = 'password123'
BUCKET_NAME = "bronze"
SEGMENT_DURATION = 20 
MODEL_PATH = './Models/face_landmarker.task'

# --- NEW CONFIGURATION CONSTANT ---
TEMP_DIR = "./temp_videos" 
os.makedirs(TEMP_DIR, exist_ok=True) # Ensures the folder exists before writing to it


# Landmark initialization
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Inicjalizacja reszty
s3_client = boto3.client('s3', endpoint_url=MINIO_URL, 
                        aws_access_key_id=MINIO_ACCESS_KEY, 
                        aws_secret_access_key=MINIO_SECRET_KEY)
producer = Producer({'bootstrap.servers': KAFKA_SERVER})

def delivery_report(err, msg):
    if err is not None: print(f"Kafka Error: {err}")

def upload_video_segment(file_path, filename):
    try:
        s3_client.upload_file(file_path, BUCKET_NAME, f"video/{STUDENT_ID}/{filename}")
        if os.path.exists(file_path): 
            os.remove(file_path)
            print(f"Successfully deleted local temp file: {file_path}")
    except Exception as e: 
        print(f"S3 Error: {e}")

# --- PĘTLA GŁÓWNA ---
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

print("Uruchamianie kamery z obsługą Blendshapes... 'q' aby wyjść.")

try:
    while cap.isOpened():
        timestamp = int(time.time())
        filename = f"vid_{timestamp}.avi"
        
        # --- UPDATED PATH PLACEMENT ---
        local_path = os.path.join(TEMP_DIR, f"temp_{filename}")
        
        out = cv2.VideoWriter(local_path, fourcc, 20.0, (640, 480))
        
        segment_end_time = time.time() + SEGMENT_DURATION
        
        while time.time() < segment_end_time:
            ret, frame = cap.read()
            if not ret: break
            
            out.write(frame)
            
            # Przetwarzanie obrazu dla MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detekcja (Nowy sposób)
            detection_result = detector.detect(mp_image)
            
            blendshapes_data = {}
            if detection_result.face_blendshapes:
                blendshapes_data = {b.category_name: round(float(b.score), 3) 
                        for b in detection_result.face_blendshapes[0]}
                
                coords = [{"x": l.x, "y": l.y, "z": l.z} for l in detection_result.face_landmarks[0]]
                
                data = {
                    "student_id": STUDENT_ID,
                    "ts": time.time(),
                    "landmarks": coords,
                    "blendshapes": blendshapes_data
                }
                
                producer.produce('face-landmarks', json.dumps(data).encode('utf-8'), callback=delivery_report)
                producer.poll(0)
            
            if blendshapes_data:
                smile = blendshapes_data.get('mouthSmileLeft', 0)
                cv2.putText(frame, f"SMILE: {int(smile*100)}%", (20, 50), 2, 1, (0,255,0), 2)

            cv2.imshow('Ingest z Blendshapes', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                os._exit(0)

        out.release()
        
        # Spawning the upload thread
        threading.Thread(target=upload_video_segment, args=(local_path, filename)).start()

except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    producer.flush()