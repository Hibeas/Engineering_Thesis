import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# ======================================================================
# File: Models/Testing_models.py
# Description: Local test harness for running a chosen emotion model live
#              with MediaPipe face landmarker and an on-screen HUD for debug.
# ======================================================================


# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================
# Toggle this path to test either your Model 1 or Model 3 checkpoint file
CHOSEN_MODEL_PATH = './Models/FANE__emotion_model.pth' 

TASK_FILE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
TASK_FILE_PATH = "face_landmarker.task"

# Strict feature mapping array (Must exactly match your model training configuration)
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
# --- 2. AUTOMATIC MEDIAPIPE ASSET DOWNLOADER ---
# ==============================================================================
if not os.path.exists(TASK_FILE_PATH):
    print(f"Downloading official MediaPipe FaceLandmarker task file (~15MB)...")
    urllib.request.urlretrieve(TASK_FILE_URL, TASK_FILE_PATH)
    print("Download complete.")

# ==============================================================================
# --- 3. DYNAMIC PYTORCH ARCHITECTURE INITIALIZATION ---
# ==============================================================================
class PyTorchEmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, num_classes):
        super(PyTorchEmotionClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, num_classes)
        )
    def forward(self, x):
        return self.network(x)

print(f"Loading weights from: {CHOSEN_MODEL_PATH}")
if not os.path.exists(CHOSEN_MODEL_PATH):
    raise FileNotFoundError(f"Missing weights target: {CHOSEN_MODEL_PATH}")

# FIX: Added weights_only=False explicitly to silence the future warning safely
checkpoint = torch.load(CHOSEN_MODEL_PATH, weights_only=False)
EMOTION_LABELS = checkpoint['classes']
input_dim = checkpoint.get('input_dim', 52)

# Automatically adapt hidden dimensions to prevent dimension mismatch errors
if "FANE" in CHOSEN_MODEL_PATH or "FANE" in CHOSEN_MODEL_PATH.upper():
    h1, h2 = 256, 128  # Model 3 layout
    print("Layout configured for Model 3 (FANE Cleaned: 256 -> 128).")
else:
    h1, h2 = 128, 64   # Model 1 layout
    print("Layout configured for Model 1 (Baseline: 128 -> 64).")

ml_model = PyTorchEmotionClassifier(input_dim=input_dim, hidden_dim_1=h1, hidden_dim_2=h2, num_classes=len(EMOTION_LABELS))
ml_model.load_state_dict(checkpoint['model_state_dict'])
ml_model.eval()

LABEL_CLEANER = {lbl: lbl.replace('face_', '').upper() for lbl in EMOTION_LABELS}

# ==============================================================================
# --- 4. NEW MEDIAPIPE TASKS LANDMARKER SETUP ---
# ==============================================================================
base_options = python.BaseOptions(model_asset_path=TASK_FILE_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True, # Tells MediaPipe to compute the 52 muscle positions
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# ==============================================================================
# --- 5. VIDEO CAPTURE LOOP ---
# ==============================================================================
cap = cv2.VideoCapture(0)
print("\nVideo stream active. Press 'q' inside the window to close testing HUD.\n")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        
        # Convert OpenCV BGR matrix format to MediaPipe's internal Image format wrapper
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run synchronous frame detection
        detection_result = detector.detect(mp_image)

        dominant_emotion = "NEUTRAL"
        confidence_score = 0.0

        # Check if any face blendshapes are extracted
        if detection_result.face_blendshapes:
            face_blendshapes = detection_result.face_blendshapes[0] # Get primary face
            
            # Map MediaPipe category objects into a fast lookup dict
            bs_dict = {category.category_name: category.score for category in face_blendshapes}
            
            # 1. Arrange values into the exact order expected by your PyTorch training configuration
            features = [float(bs_dict.get(name, 0.0)) for name in BLENDSHAPE_NAMES]
            
            # 2. Reshape into tensor structure (1, 52)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # 3. Compute neural network inference forward pass
            with torch.no_grad():
                raw_logits = ml_model(features_tensor)
                probabilities = F.softmax(raw_logits, dim=1).numpy()[0]
            
            # 4. Map probabilities to clean text labels
            emotions_breakdown = {
                LABEL_CLEANER[EMOTION_LABELS[i]]: float(probabilities[i])
                for i in range(len(EMOTION_LABELS))
            }
            
            dominant_raw_label = max(emotions_breakdown, key=emotions_breakdown.get)
            raw_confidence = emotions_breakdown[dominant_raw_label]
            
            # Baseline dynamic threshold evaluation
            if raw_confidence > 0.25:
                dominant_emotion = dominant_raw_label
                confidence_score = raw_confidence
            else:
                dominant_emotion = "NEUTRAL"
                confidence_score = 1.0 - raw_confidence

        # ==============================================================================
        # --- 6. VISUAL HUD OVERLAY RENDERING ---
        # ==============================================================================
        # Background canvas header accent container block
        cv2.rectangle(frame, (10, 10), (380, 85), (20, 20, 20), -1)
        
        if dominant_emotion == "HAPPY":
            text_color = (100, 255, 100)  # Green
        elif dominant_emotion in ["ANGRY", "SAD", "DISGUST"]:
            text_color = (100, 100, 255)  # Red
        else:
            text_color = (255, 215, 0)    # Gold / Yellow
            
        cv2.putText(frame, f"EMOTION: {dominant_emotion}", (20, 42), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"CONFIDENCE: {confidence_score*100:.1f}%", (20, 72), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.imshow('Real-Time Model Testing HUD Interface', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n Halting tracking execution loop...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Video pipeline tracking systems offline.")