import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ==============================================================================
# --- 1. CONFIGURATION & DATA CLEANING ---
# ==============================================================================
INPUT_CSV = './Models/Training/FANE_blendshapes.csv'            # Your generated raw FANE CSV file
MODEL_SAVE_PATH = './Models/FANE__emotion_model.pth'

# Define only the 6 core high-activation classes (Stripping 'shy', 'confused', 'fear')
ALLOWED_EMOTIONS = ['happy', 'neutral', 'sad', 'surprise', 'angry', 'disgust']

print(f"🔄 Loading feature data from: {INPUT_CSV}...")
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Could not find '{INPUT_CSV}'. Ensure it is in this directory.")

df_raw = pd.read_csv(INPUT_CSV)
print(f"📊 Original dataset size: {df_raw.shape[0]} rows.")

# Step A: Filter out weak, overlapping classes
df_clean = df_raw[df_raw['label'].isin(ALLOWED_EMOTIONS)].reset_index(drop=True)
print(f"🧹 Cleaned dataset size (6 core emotions): {df_clean.shape[0]} rows.")

# Step B: Split features (X) and text targets (y)
X = df_clean.drop(columns=['label']).values  # Shape: (N, 52)
y_text = df_clean['label'].values

# Step C: Encode textual targets to integers (0, 1, 2...)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_text)
num_classes = len(encoder.classes_)
print(f"🏷️ Encoded target classes: {encoder.classes_.tolist()}")

# Step D: Train/Test Split (80% Train, 20% Evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Step E: Convert numpy blocks into PyTorch Tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

# ==============================================================================
# --- 2. NETWORK ARCHITECTURE (MODEL 3 WIDE SPECIFICATION) ---
# ==============================================================================
class FaneCleanedEmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FaneCleanedEmotionClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),  # Expanded hidden capacity from 128 to 256
            nn.ReLU(),
            nn.Dropout(0.3),            # Strong regularization to handle overfitting
            nn.Linear(256, 128),        # Secondary hidden layer
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

# Initialize network model
model = FaneCleanedEmotionClassifier(input_dim=52, num_classes=num_classes)

# ==============================================================================
# --- 3. OPTIMIZATION LOOP (300 EPOCHS) ---
# ==============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 300

print("\n🚀 Commencing Model 3 Training Pipeline...")
model.train()

for epoch in range(epochs):
    # Forward Pass
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    
    # Backward Pass & Weights Update
    loss.backward()
    optimizer.step()
    
    # Log progress every 50 epochs
    if (epoch + 1) % 50 == 0 or epoch == 0:
        # Calculate training accuracy on the fly
        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y_train_t).sum().item()
            train_acc = (correct / y_train_t.size(0)) * 100
        print(f"Epoch [{epoch+1:03d}/{epochs}] ── Loss: {loss.item():.4f} ── Train Accuracy: {train_acc:.2f}%")

# ==============================================================================
# --- 4. PIPELINE EVALUATION ---
# ==============================================================================
print("\n🧪 Running evaluation against test split data...")
model.eval()

with torch.no_grad():
    test_outputs = model(X_test_t)
    loss_test = criterion(test_outputs, y_test_t)
    _, predicted_test = torch.max(test_outputs, 1)
    
    # Map numerical categories back to original readable text labels
    y_test_labels = encoder.inverse_transform(y_test)
    predicted_labels = encoder.inverse_transform(predicted_test.numpy())
    
    print("\n" + "="*60)
    print("📋 MODEL 3 FINAL CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test_labels, predicted_labels, digits=4))
    print("="*60)

# ==============================================================================
# --- 5. CHECKPOINT PERSISTENCE ---
# ==============================================================================
# We save structural context parameters inside the state dictionary file
# so the Kafka consumer script can dynamically read target labels and layouts.
torch.save({
    'model_state_dict': model.state_dict(),
    'classes': encoder.classes_.tolist(),
    'input_dim': 52
}, MODEL_SAVE_PATH)

print(f"\n💾 Optimization complete! Weights safely stored to: '{MODEL_SAVE_PATH}'\n")