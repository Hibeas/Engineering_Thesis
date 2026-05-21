import os
import csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup MediaPipe
base_options = python.BaseOptions(model_asset_path='./Models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Configuration - FANE Model 3
#DATASET_DIR = r"C:\Users\Hibeas\Desktop\project_ai\My_Model\fan\fane_data" 
#OUTPUT_CSV = "./Models/Training/FAN_blendshapes.csv"

# Configuration - AffectNet Model 1
DATASET_DIR = r"C:\Users\Hibeas\archive\images" 
OUTPUT_CSV = "./Models/Training/AffectNet_blendshapes.csv"

# Predefined order of all 52 shapes to guarantee index consistency
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

with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    # Header: blendshapes names + the target emotion label
    writer.writerow(BLENDSHAPE_NAMES + ['label'])
    
    # Loop over folders: happy, sad, angry, neutral, etc.
    for emotion_label in os.listdir(DATASET_DIR):
        emotion_path = os.path.join(DATASET_DIR, emotion_label)
        if not os.path.isdir(emotion_path): continue
        
        print(f"Processing emotion group: {emotion_label}")
        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            
            # Load image and convert to MediaPipe Format
            image = cv2.imread(img_path)
            if image is None: continue
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            
            detection_result = detector.detect(mp_image)
            
            if detection_result.face_blendshapes:
                # Extract scores
                scores = {b.category_name: b.score for b in detection_result.face_blendshapes[0]}
                row = [scores.get(name, 0.0) for name in BLENDSHAPE_NAMES]
                row.append(emotion_label)
                writer.writerow(row)