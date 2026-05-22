# Real-Time Student Emotion Monitoring Infrastructure

A decoupled, real-time data streaming platform designed to capture facial expressions, extract normalized geometric blendshapes, and process emotional states during laboratory sessions. This project serves as an engineering thesis implementation utilizing an enterprise-grade Data Lakehouse architecture (Bronze to Silver layers) powered by Apache Kafka, Google MediaPipe, and PyTorch.

---

## System Architecture

The repository is structurally decoupled into two distinct pipelines to optimize edge compute and minimize network bandwidth overhead:

1. **Ingestion & Feature Extraction (Bronze Layer):** Captures local webcam frames at the student terminal, extracts 52 muscle-activation coefficients (blendshapes) using Google MediaPipe Tasks API, and streams lightweight JSON packets to Kafka over an SSH tunnel.
2. **Analytical & Neural Inference (Silver Layer):** Centrally consumes the blendshape stream from Kafka, runs deep learning inference passes via PyTorch, and broadcasts classified, high-confidence target emotions downstream.

## Modular Design: Why a Two-File Architecture?

Rather than maintaining a single monolithic script that handles everything locally, this platform strictly splits execution into a **Client-Side Producer (`Bronze_landmarks.py`)** and a **Server-Side Consumer (`Silver_FANE.py` / `Silver_AffectNet.py`)**. 

This intentional separation of concerns was chosen for several critical data engineering and DevOps reasons:

### 1. Seamless, Zero-Downtime Model Hot-Swapping
* **The Monolithic Problem:** If the machine learning model ran directly inside a single file on the student's computer, deploying an upgraded or retrained neural network would require manually pushing updates and stopping execution on every single terminal in the computer lab.
* **The Modular Solution:** With a two-file system, the client script only cares about extracting and streaming standard facial coordinates. The machine learning model is isolated completely on the central server. If you want to switch from Model 1 to Model 3, or fix a bug in the inference logic, you modify and restart **one single file on the server** without ever touching or disrupting the active client terminals.

### 2. Centralized, Server-Side Computational Load
* Running deep learning matrix multiplications dynamically on a live camera stream requires consistent hardware resources. 
* Moving the PyTorch inference pass to the central Silver server prevents performance degradation on older lab desktops. It ensures the client application remains extremely lightweight, leaving the student's local CPU/GPU entirely free to run heavy IDEs, compilation tasks, or laboratory code.

### 3. Clean Codebase and Separation of Concerns
* **Client File:** Dedicated purely to edge I/O hardware management, frame mirroring, and streaming transport protocols.
* **Server File:** Dedicated purely to downstream data validation, tensor parsing, multi-class neural network forwarding, confidence filtering, and enterprise analytics formatting.
* This clean decoupling allows for independent testing, debugging, and scaling of the ingestion and processing layers.

## Repository Structure

Based on the project root development environment:

```text
ENGINEERING_THESIS/
├── Models/
│   ├── Training/
│   │   ├── AffectNet_blendshapes.csv   # Model 1 extracted training dataset
│   │   ├── FANE_blendshapes.csv        # Model 3 raw training dataset
│   │   ├── Extract_feature.py          # MediaPipe feature extraction script
│   │   ├── train_model_AffectNet.py    # Training script for Model 1 (Baseline)
│   │   ├── train_model_FANE.py         # Training script for Model 3 (Optimized)
│   │   └── Training_data_info.txt      # Structural logs and dataset annotations
│   │
│   ├── AffectNet_blendshapes.pth       # Serialized weights for Model 1
│   ├── FANE_emotion_model.pth          # Serialized weights for Model 3
│   ├── face_landmarker.task            # Google MediaPipe configuration binary
│   ├── Model_info.txt                  # Model performance metrics and hidden layer data
│   └── Testing_models.py               # Standalone live webcam local test harness
│
├── temp_videos/                        # Cache directory for sample capture validation
├── Bronze_landmarks.py                 # Kafka Producer: Edge capture & blendshape ingestion
├── Silver_AffectNet.py                 # Kafka Consumer/Producer: Model 1 Inference execution
├── Silver_calculating.py               # Kafka Consumer/Producer: Rule-based Expert System script
├── Silver_FANE.py                      # Kafka Consumer/Producer: Model 3 Inference execution
└── Usefull_comands.txt                 # Shell commands for Kafka brokers and deployment
