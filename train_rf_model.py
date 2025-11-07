# ------------------------------------------------------
# voc_nn_train.py — Neural Network Model for VOC Prediction
# ------------------------------------------------------

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------
df = pd.read_csv("data/synthetic_voc_dataset.csv")
print("✅ Dataset loaded:", df.shape)

X = df.drop(columns=["label"])
y = df["label"]

# Encode string labels into integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save encoder for backend decoding
os.makedirs("model", exist_ok=True)
joblib.dump(le, "model/label_encoder.pkl")

# ------------------------------------------------------
# 2. Split Data (80/20)
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ------------------------------------------------------
# 3. Scale Inputs
# ------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "model/voc_scaler.pkl")

# ------------------------------------------------------
# 4. Build Neural Network Model
# ------------------------------------------------------
num_features = X_train.shape[1]
num_classes = len(np.unique(y_encoded))

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(num_features,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------------------------------------------
# 5. Train Model
# ------------------------------------------------------
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ------------------------------------------------------
# 6. Evaluate
# ------------------------------------------------------
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

try:
    roc_auc = roc_auc_score(tf.keras.utils.to_categorical(y_test), y_pred_probs)
    print(f"\n🔥 ROC-AUC: {roc_auc:.4f}")
except Exception:
    pass

# ------------------------------------------------------
# 7. Save Model
# ------------------------------------------------------
model.save("model/voc_nn_model.h5")
print("\n💾 Saved model to model/voc_nn_model.h5")

# ------------------------------------------------------
# 8. Plot training progress (optional)
# ------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("model/training_curve.png")
plt.show()
