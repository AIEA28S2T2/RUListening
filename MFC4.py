import os
import random
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.schedules import CosineDecay
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from collections import Counter
import tensorflow as tf
from tensorflow.keras import backend as K

# Disable TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initial parameters
epochs = 64
batch_size = 32  # Reduced batch size for better stability
img_dims = (96, 96, 3)
lr = 1e-4  # Learning rate

data = []
labels = []

# Load images from dataset
image_files = [f for f in glob.glob(r'D:\EOC_MFC_New\Images' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

# Convert images to arrays and assign labels
for img in image_files:
    image = cv2.imread(img)
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2]
    labels.append(1 if label == "Eyes Closed" else 0)  # 1 = Not Attentive, 0 = Attentive

# Pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Print dataset label distribution
print("Label distribution before balancing:", Counter(labels.flatten()))

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels.flatten())
class_weight_dict = dict(enumerate(class_weights))

# Split dataset
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# Augmenting dataset with enhanced strategies
aug = ImageDataGenerator(
    rotation_range=30,  # Increased range
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.4,  # Higher shearing
    zoom_range=0.4,  # Increased zoom
    horizontal_flip=True,
    fill_mode="nearest",
    brightness_range=[0.4, 1.6]  # Stronger variations
)

# Define Focal Loss
def focal_loss(alpha=0.5, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        
        bce = K.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        loss = alpha * K.pow(1 - p_t, gamma) * bce
        return K.mean(loss)
    
    return loss

# Define model using EfficientNetB0
def build(width, height, depth, classes):
    base_model = EfficientNetB0(input_shape=(height, width, depth), include_top=False, weights="imagenet")
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu"),
        Dropout(0.4),
        Dense(classes, activation="softmax")
    ])
    return model

# Build model
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=2)

# Compile model with Cosine Annealing Learning Rate
lr_schedule = CosineDecay(initial_learning_rate=lr, decay_steps=epochs * len(trainX) // batch_size)
opt = Adam(learning_rate=lr_schedule)
model.compile(loss=focal_loss(alpha=0.5, gamma=2), optimizer=opt, metrics=["accuracy"])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
H = model.fit(
    aug.flow(trainX, trainY, batch_size=batch_size),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // batch_size,
    epochs=epochs,
    verbose=1,
    class_weight=class_weight_dict,
    callbacks=[early_stopping]
)

# Save trained model
model.save('MFC_EOC_PROJECT_OPTIMIZED.keras')

# Plot training and validation accuracy/loss
plt.figure()
plt.plot(H.history['accuracy'], label='Train Accuracy')
plt.plot(H.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

plt.figure()
plt.plot(H.history['loss'], label='Train Loss')
plt.plot(H.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()