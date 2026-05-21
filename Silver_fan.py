import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from confluent_kafka import Consumer, Producer

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================
CHOSEN_MODEL_FILE = './Models/FANE_cleaned_emotion_model.pth' # Your optimized Model 3 file

KAFKA_SERVER = 'localhost:8081'
INPUT_TOPIC = 'face-landmarks'
OUTPUT_TOPIC = 'face-emotions'

# Strict order of all 52 blendshapes (Must match the training/extraction script exactly)
BLENDSHAPE_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
    "noseSneerRight", "tongueOut"
]

# ==============================================================================
# --- 2. PYTORCH ARCHITECTURE (MODEL 3 SPECIFIC: 256 -> 128) ---
# ==============================================================================
class FaneCleanedEmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FaneCleanedEmotionClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),  # Widened layer for Model 3
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),         # Secondary hidden layer
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.network(x)

# Load checkpoint and verify file validity
print(f"🔄 Initializing Neural Network using Model 3 (FANE Cleaned) weights: {CHOSEN_MODEL_FILE}...")
if not os.path.exists(CHOSEN_MODEL_FILE):
    print(f"❌ Critical Error: Weights file '{CHOSEN_MODEL_FILE}' not found!")
    exit(1)

checkpoint = torch.load(CHOSEN_MODEL_FILE)
EMOTION_LABELS = checkpoint['classes']  # Contains the 6 filtered classes
input_dim = checkpoint.get('input_dim', 52)

# Build Model 3 network layout
ml_model = FaneCleanedEmotionClassifier(input_dim=input_dim, num_classes=len(EMOTION_LABELS))
ml_model.load_state_dict(checkpoint['model_state_dict'])
ml_model.eval()

# Maps internal text classes safely (Forces uppercase representation for the Silver payload)
LABEL_CLEANER = {lbl: lbl.replace('face_', '').upper() for lbl in EMOTION_LABELS}
print(f"✅ Model 3 Ready! Active tracking targets: {[LABEL_CLEANER[l] for l in EMOTION_LABELS]}")

# ==============================================================================
# --- 3. INFRASTRUCTURE INITIALIZATION (KAFKA COUPLING) ---
# ==============================================================================
consumer_conf = {
    'bootstrap.servers': KAFKA_SERVER,
    'group.id': 'silver-fane-inference-v1',
    'auto.offset.reset': 'earliest'
}

producer_conf = {'bootstrap.servers': KAFKA_SERVER}

consumer = Consumer(consumer_conf)
consumer.subscribe([INPUT_TOPIC])
producer = Producer(producer_conf)

def delivery_report(err, msg):
    if err is not None:
        print(f"❌ Kafka Streaming Delivery failed: {err}")

# ==============================================================================
# --- 4. STREAM PROCESSING LOOP ---
# ==============================================================================
print(f"🚀 Real-Time FANE Model 3 Silver Engine Online.")
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
            # Decode incoming JSON data block from Bronze Layer
            raw_payload = json.loads(msg.value().decode('utf-8'))
            bs_dict = raw_payload.get('blendshapes', {})
            
            if not bs_dict:
                continue

            # 1. Map raw json properties cleanly into the strict 52-feature layout
            features = [float(bs_dict.get(name, 0.0)) for name in BLENDSHAPE_NAMES]
            
            # 2. Shape to dimensions expected by PyTorch batch loader: (1, 52)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # 3. Compute inference pass
            with torch.no_grad():
                raw_logits = ml_model(features_tensor)
                probabilities = F.softmax(raw_logits, dim=1).numpy()[0]
            
            # 4. Construct response dictionary for all 6 target emotions
            emotions_breakdown = {
                LABEL_CLEANER[EMOTION_LABELS[i]]: round(float(probabilities[i]), 3)
                for i in range(len(EMOTION_LABELS))
            }
            
            # 5. Extract dominant emotional metadata values
            dominant_raw_label = max(emotions_breakdown, key=emotions_breakdown.get)
            confidence_score = emotions_breakdown[dominant_raw_label]
            
            # Standard confidence check logic
            if confidence_score > 0.25:
                dominant_emotion = dominant_raw_label
            else:
                dominant_emotion = "NEUTRAL"

            # 6. Build out the structured Silver Layer event model payload
            silver_payload = {
                "student_id": raw_payload.get("student_id", "unknown"),
                "ts": int(raw_payload.get("ts", time.time()) * 1000), # Standardize to Unix Epoch ms
                "emotions": emotions_breakdown,                     # Complete probability vector
                "dominant": dominant_emotion,                        # High certainty target tag
                "confidence": confidence_score                       # Value indicator
            }

            # 7. Broadcast the processed event payload out to the Silver analytical topic
            producer.produce(
                OUTPUT_TOPIC, 
                json.dumps(silver_payload).encode('utf-8'),
                callback=delivery_report
            )
            producer.poll(0)

            print(f"Stream -> Student: {silver_payload['student_id']} | Predict: {dominant_emotion:<10} ({confidence_score:.2f})")

        except Exception as e:
            print(f"Pipeline running processing exception: {e}")

except KeyboardInterrupt:
    print("\nShutting down FANE inference stream gracefully...")
finally:
    producer.flush()
    consumer.close()
    print("Offline.")