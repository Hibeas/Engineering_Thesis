import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ==============================================================================
# --- 1. CONFIGURATION & DATA INGESTION ---
# ==============================================================================
INPUT_CSV = './Models/Training/AffectNet_blendshapes.csv'       # Your baseline Blendshapes dataset
MODEL_SAVE_PATH = './Models/AffectNet_blendshapes.pth'     # Target weights file name

print(f"Loading baseline features from: {INPUT_CSV}...")
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Could not find '{INPUT_CSV}'. Ensure it is in your root directory.")

df = pd.read_csv(INPUT_CSV)
print(f"Dataset successfully parsed: {df.shape[0]} rows containing {df.shape[1] - 1} features.")

# Split features (X) and textual classes (y)
X = df.drop(columns=['label']).values          # Extracted 52 MediaPipe blendshape channels
y_text = df['label'].values

# Encode string labels into numerical categories (0, 1, 2...)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_text)
num_classes = len(encoder.classes_)
print(f"Target categories mapped ({num_classes}): {encoder.classes_.tolist()}")

# Train/Test Split (80% Training pool, 20% validation pool)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Convert arrays into native PyTorch Float/Long Tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

# ==============================================================================
# --- 2. NETWORK ARCHITECTURE (MODEL 1 SPECIFICATION) ---
# ==============================================================================
class BaselineEmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BaselineEmotionClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),  # Baseline hidden dimension 1
            nn.ReLU(),
            nn.Dropout(0.3),            # Prevents structural node overfitting
            nn.Linear(128, 64),         # Baseline hidden dimension 2
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

# Initialize Model 1 structure
model = BaselineEmotionClassifier(input_dim=52, num_classes=num_classes)

# ==============================================================================
# --- 3. OPTIMIZATION TRACK (100 EPOCHS) ---
# ==============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100

print("\nStarting Model 1 benchmark training...")
model.train()

for epoch in range(epochs):
    # Forward Pass
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    
    # Backward Pass & Weights Propagation
    loss.backward()
    optimizer.step()
    
    # Log progress every 20 epochs to keep trace clean
    if (epoch + 1) % 20 == 0 or epoch == 0:
        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y_train_t).sum().item()
            train_acc = (correct / y_train_t.size(0)) * 100
        print(f"Epoch [{epoch+1:03d}/{epochs}] ── Error Loss: {loss.item():.4f} ── Accuracy: {train_acc:.2f}%")

# ==============================================================================
# --- 4. VALIDATION PERFORMANCE ---
# ==============================================================================
print("\n🧪 Evaluating against test split parameters...")
model.eval()

with torch.no_grad():
    test_outputs = model(X_test_t)
    _, predicted_test = torch.max(test_outputs, 1)
    
    # Invert integer labels back to readable text values
    y_test_labels = encoder.inverse_transform(y_test)
    predicted_labels = encoder.inverse_transform(predicted_test.numpy())
    
    print("\n" + "="*60)
    print("MODEL 1 INITIAL CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test_labels, predicted_labels, digits=4))
    print("="*60)

# ==============================================================================
# --- 5. SYSTEM CHECKPOINT EXPORT ---
# ==============================================================================
# Saving labels along with the state dictionary ensures your Kafka Silver 
# streaming scripts can load weights safely without needing a static hardcoded map.
torch.save({
    'model_state_dict': model.state_dict(),
    'classes': encoder.classes_.tolist(),
    'input_dim': 52
}, MODEL_SAVE_PATH)

print(f"\nValidation complete. Model 1 weights saved to: '{MODEL_SAVE_PATH}'\n")