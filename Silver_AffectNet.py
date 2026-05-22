import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from confluent_kafka import Consumer, Producer

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================
CHOSEN_MODEL_FILE = './Models/kaggle_landmarks_emotion_model.pth' # Your trained Model 1 file

KAFKA_SERVER = 'localhost:8081'
INPUT_TOPIC = 'face-landmarks'
OUTPUT_TOPIC = 'face-emotions'

# ==============================================================================
# --- 2. LANDMARK NORMALIZATION FUNCTION ---
# ==============================================================================
def normalize_landmarks(landmarks_list):
    """
    Transforms raw x, y, z camera coordinates into a translation-invariant
    and scale-invariant feature vector.
    """
    # Convert list of dicts [{"x":..., "y":..., "z":...}] to a numpy array (478, 3)
    coords = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks_list])
    
    # 1. Translation Invariance: Center the face around the nose bridge (Landmark 4)
    nose_bridge = coords[4]
    centered_coords = coords - nose_bridge
    
    # 2. Scale Invariance: Calculate face bounding scale using distance 
    # between forehead (Landmark 10) and chin (Landmark 152)
    face_height = np.linalg.norm(coords[10] - coords[152])
    if face_height == 0:
        face_height = 1.0 # Prevent division by zero
        
    normalized_coords = centered_coords / face_height
    
    # Flatten from (478, 3) to a single 1D vector of 1434 features
    return normalized_coords.flatten()

# ==============================================================================
# --- 3. PYTORCH ARCHITECTURE (MODEL 1 SPECIFIC: 128 -> 64) ---
# ==============================================================================
class LandmarksEmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LandmarksEmotionClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),  # Model 1 architecture baseline
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.network(x)

# Load checkpoint and extract dimensions dynamically
print(f"Initializing Neural Network using Landmark weights: {CHOSEN_MODEL_FILE}...")
if not os.path.exists(CHOSEN_MODEL_FILE):
    print(f"Critical Error: Model file '{CHOSEN_MODEL_FILE}' not found!")
    exit(1)

checkpoint = torch.load(CHOSEN_MODEL_FILE)
EMOTION_LABELS = checkpoint['classes']
# For 478 3D points, input_dim will be 478 * 3 = 1434
input_dim = checkpoint.get('input_dim', 1434) 

# Instantiate and build Model 1
ml_model = LandmarksEmotionClassifier(input_dim=input_dim, num_classes=len(EMOTION_LABELS))
ml_model.load_state_dict(checkpoint['model_state_dict'])
ml_model.eval()

# Clean up raw labels (e.g., 'face_happy' -> 'HAPPY')
LABEL_CLEANER = {lbl: lbl.replace('face_', '').upper() for lbl in EMOTION_LABELS}
print(f"Model 1 Ready! Target classes: {[LABEL_CLEANER[l] for l in EMOTION_LABELS]}")

# ==============================================================================
# --- 4. INFRASTRUCTURE INITIALIZATION (KAFKA COUPLING) ---
# ==============================================================================
consumer_conf = {
    'bootstrap.servers': KAFKA_SERVER,
    'group.id': 'silver-landmarks-inference-v1',
    'auto.offset.reset': 'earliest'
}

producer_conf = {'bootstrap.servers': KAFKA_SERVER}

consumer = Consumer(consumer_conf)
consumer.subscribe([INPUT_TOPIC])
producer = Producer(producer_conf)

def delivery_report(err, msg):
    if err is not None:
        print(f"Kafka Streaming Delivery failed: {err}")

# ==============================================================================
# --- 5. PROCESSING PIPELINE ---
# ==============================================================================
print(f"🚀 Real-Time Landmark Silver Engine Online.")
print(f"📥 Reading: {INPUT_TOPIC} | 📤 Writing: {OUTPUT_TOPIC}\n")

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None: 
            continue
        if msg.error():
            print(f"Kafka Ingestion Error: {msg.error()}")
            continue

        try:
            # Decode payload from Bronze stream
            raw_payload = json.loads(msg.value().decode('utf-8'))
            landmarks_list = raw_payload.get('landmarks', [])
            
            # Ensure we have the full set of MediaPipe mesh points
            if not landmarks_list or len(landmarks_list) < 478:
                continue

            # 1. Normalize the raw landmarks mathematically
            flat_features = normalize_landmarks(landmarks_list)
            
            # 2. Convert to PyTorch tensor format and shape to (1, 1434)
            features_tensor = torch.FloatTensor(flat_features).unsqueeze(0)
            
            # 3. Compute model inference
            with torch.no_grad():
                raw_logits = ml_model(features_tensor)
                probabilities = F.softmax(raw_logits, dim=1).numpy()[0]
            
            # 4. Construct response dictionary containing entire probability space
            emotions_breakdown = {
                LABEL_CLEANER[EMOTION_LABELS[i]]: round(float(probabilities[i]), 3)
                for i in range(len(EMOTION_LABELS))
            }
            
            # 5. Extract dominant metrics
            dominant_raw_label = max(emotions_breakdown, key=emotions_breakdown.get)
            confidence_score = emotions_breakdown[dominant_raw_label]
            
            # Threshold fallback logic matching your baseline criteria
            if confidence_score > 0.25:
                dominant_emotion = dominant_raw_label
            else:
                dominant_emotion = "NEUTRAL"

            # 6. Assemble Silver structured schema payload
            silver_payload = {
                "student_id": raw_payload.get("student_id", "unknown"),
                "ts": int(raw_payload.get("ts", time.time()) * 1000), # Milliseconds standard
                "emotions": emotions_breakdown,                     # Distribution breakdown
                "dominant": dominant_emotion,                        # Predicted target
                "confidence": confidence_score                       # Certainty metric
            }

            # 7. Push metrics back out onto Kafka Silver Layer
            producer.produce(
                OUTPUT_TOPIC, 
                json.dumps(silver_payload).encode('utf-8'),
                callback=delivery_report
            )
            producer.poll(0)

            print(f"Stream -> Student: {silver_payload['student_id']} | Predict: {dominant_emotion:<10} ({confidence_score:.2f})")

        except Exception as e:
            print(f"Pipeline processing error: {e}")

except KeyboardInterrupt:
    print("\nShutting down landmark inference stream gracefully...")
finally:
    producer.flush()
    consumer.close()
    print("Offline.")