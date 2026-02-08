import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------
# CONFIG
# -----------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
BASE_DIR = "wheat-disease-dataset-small"
TEST_DIR = os.path.join(BASE_DIR, "test")

# -----------------------
# LOAD MODEL
# -----------------------
model = load_model("wheat_disease_model.h5")
print("Model loaded successfully")

# -----------------------
# LOAD TEST DATA
# -----------------------
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# -----------------------
# PREDICTIONS
# -----------------------
pred_probs = model.predict(test_data)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_data.classes

class_names = list(test_data.class_indices.keys())

# -----------------------
# CLASSIFICATION REPORT
# -----------------------
print("\nCLASSIFICATION REPORT\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# -----------------------
# CONFUSION MATRIX
# -----------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()