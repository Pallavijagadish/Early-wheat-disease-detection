# wheat_disease_efficientnet.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ==============================
# STEP 1: DATASET PATH
# ==============================

train_dir = r"C:\Users\MY PC\Desktop\Early-wheat-disease-detection\Wheat_Disease\train"
val_dir = r"C:\Users\MY PC\Desktop\Early-wheat-disease-detection\Wheat_Disease\validation"

# ==============================
# STEP 2: IMAGE SETTINGS
# ==============================

img_size = (224, 224)
batch_size = 32

# ==============================
# STEP 3: DATA GENERATORS
# ==============================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

print("Classes:", train_generator.class_indices)

# ==============================
# STEP 4: MODEL (EfficientNetB0)
# ==============================

base_model = EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# STEP 5: TRAIN MODEL
# ==============================

epochs = 10

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# ==============================
# STEP 6: PLOT GRAPHS
# ==============================

# Accuracy Graph
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# Loss Graph
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Graph")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# ==============================
# STEP 7: SAVE MODEL
# ==============================

model.save("wheat_disease_efficientnet_model.h5")

print("✅ Model training completed and saved successfully!")