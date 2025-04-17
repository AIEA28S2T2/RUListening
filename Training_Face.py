import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from collections import Counter
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Parameters - updated values
EPOCHS = 50  # Increased epochs with early stopping
BATCH_SIZE = 16  # Smaller batch size for better generalization
IMG_DIMS = (224, 224, 3)  # Optimal for EfficientNet
INITIAL_LR = 1e-4  # Higher learning rate with decay

# Single data directory (using only what was previously the training directory)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'Dataset_Face')

'''
# Single data directory (using only what was previously the training directory)
DATA_DIR = r'C:\Users\aamit\Downloads\Niru_CNN\Final\Dataset_Face'
'''

# Class definitions - based on your folders
CLASS_NAMES = ["Forward", "Left Mirror", "Radio", "Rearview", "Speedometer", "Lap"]

# Attentiveness mapping
ATTENTIVE_CLASSES = ["Forward", "Rearview", "Radio", "Speedometer"]
NON_ATTENTIVE_CLASSES = ["Left Mirror", "Lap"]

def load_dataset(directory):
    global CLASS_NAMES
    data = []
    labels = []
    skipped_classes = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path):
            print(f"Warning: Missing class directory - {class_name}")
            skipped_classes.append(class_name)
            continue

        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(glob.glob(os.path.join(class_path, ext)))

        print(f"Loaded {len(image_paths)} images from {class_name}")

        for img_path in image_paths:
            try:
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMG_DIMS[:2])
                
                # Preprocessing for EfficientNet
                image = tf.keras.applications.efficientnet.preprocess_input(image)
                
                image = img_to_array(image)
                data.append(image)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
    
    # Update CLASS_NAMES by removing skipped classes
    
    if skipped_classes:
        CLASS_NAMES = [cls for cls in CLASS_NAMES if cls not in skipped_classes]
        print(f"Updated CLASS_NAMES: {CLASS_NAMES}")

    return np.array(data), np.array(labels)

# Load all data from a single directory
print("Loading data...")
all_data, all_labels = load_dataset(DATA_DIR)
if len(all_data) == 0:
    raise ValueError("No data loaded - check directory paths")

print("\nDataset Summary:")
print(f"Total samples: {len(all_data)}")
print("Class distribution:", Counter(all_labels))

# Recompute labels if CLASS_NAMES changed
if len(np.unique(all_labels)) != len(CLASS_NAMES):
    # Map old indices to new indices
    old_to_new = {}
    current_idx = 0
    for i in range(max(all_labels)+1):
        if i in all_labels:
            old_to_new[i] = current_idx
            current_idx += 1
    
    # Apply mapping
    all_labels = np.array([old_to_new[label] for label in all_labels])
    print("Remapped labels distribution:", Counter(all_labels))

# First split: 80% training+validation, 20% test
train_val_data, test_data, train_val_labels, test_labels = train_test_split(
    all_data, all_labels,
    test_size=0.2,
    random_state=42,
    stratify=all_labels
)

# Second split: From the 80%, use 75% for training and 25% for validation (0.75 * 0.8 = 0.6 or 60% of original data)
train_data, val_data, train_labels, val_labels = train_test_split(
    train_val_data, train_val_labels,
    test_size=0.25,  # 25% of 80% = 20% of original data
    random_state=42,
    stratify=train_val_labels
)

print("\nTrain-Validation-Test Split:")
print(f"Training samples: {len(train_data)} ({len(train_data)/len(all_data)*100:.1f}%)")
print(f"Validation samples: {len(val_data)} ({len(val_data)/len(all_data)*100:.1f}%)")
print(f"Test samples: {len(test_data)} ({len(test_data)/len(all_data)*100:.1f}%)")

# One-hot encode labels
train_labels_onehot = to_categorical(train_labels, num_classes=len(CLASS_NAMES))
val_labels_onehot = to_categorical(val_labels, num_classes=len(CLASS_NAMES))
test_labels_onehot = to_categorical(test_labels, num_classes=len(CLASS_NAMES))

# Class weights to handle imbalance - more aggressive weighting
class_counts = Counter(train_labels)
total_samples = len(train_labels)
class_weights = {}

for i, count in class_counts.items():
    # More aggressive weighting to help minority classes
    weight = (1 / count) * (total_samples / len(class_counts)) * 1.5
    class_weights[i] = weight

# Normalize weights to be centered around 1.0
weight_sum = sum(class_weights.values())
for i in class_weights:
    class_weights[i] = class_weights[i] * len(class_weights) / weight_sum

print("\nClass weights:", class_weights)

# Enhanced Data Augmentation
train_aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

# Build EfficientNetB0-based model
def build_model():
    # Use EfficientNetB0 as the base model
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=IMG_DIMS)
    
    # Partial fine-tuning: freeze early layers but train later layers
    for layer in base_model.layers[:-20]:  # Freeze all but the last 20 layers
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)  # Increased dropout for better regularization
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(len(CLASS_NAMES), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

# Build the model
model = build_model()

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=INITIAL_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc')]
)

# Display model summary
model.summary()

# Enhanced callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7),
    ModelCheckpoint('Face.keras', save_best_only=True, 
                    monitor='val_accuracy', verbose=1)
]

# Train model with data augmentation
print("\nTraining...")
history = model.fit(
    train_aug.flow(train_data, train_labels_onehot, batch_size=BATCH_SIZE),
    validation_data=(val_data, val_labels_onehot),
    steps_per_epoch=len(train_data)//BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# Evaluate on validation set
val_loss, val_acc, val_prec, val_rec, val_auc = model.evaluate(val_data, val_labels_onehot, verbose=1)
print(f"\nValidation Accuracy: {val_acc:.4f}")
print(f"Validation Precision: {val_prec:.4f}")
print(f"Validation Recall: {val_rec:.4f}")
print(f"Validation AUC: {val_auc:.4f}")

# Evaluate on test set
test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(test_data, test_labels_onehot, verbose=1)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_prec:.4f}")
print(f"Test Recall: {test_rec:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Plot training history
def plot_training(history):
    metrics = ['accuracy', 'loss', 'precision', 'recall', 'auc']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        if metric in history.history:
            plt.subplot(2, 3, i+1)
            plt.plot(history.history[metric], label=f'Train {metric}')
            plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
            plt.title(f'{metric.capitalize()}', fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_curves_efficientnet.png", dpi=300)
    plt.show()

plot_training(history)

# Generate predictions and confusion matrix
Y_pred = model.predict(test_data)
y_pred_classes = np.argmax(Y_pred, axis=1)
y_true_classes = np.argmax(test_labels_onehot, axis=1)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix', fontweight='bold', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# Print classification report
print("\nClassification Report:")
cr = classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES)
print(cr)

# Create attentiveness classifier
def map_to_attentiveness(class_index):
    class_name = CLASS_NAMES[class_index]
    if class_name in ATTENTIVE_CLASSES:
        return "Attentive"
    else:
        return "Non-Attentive"

# Map predictions to attentiveness
y_true_attentiveness = [map_to_attentiveness(idx) for idx in y_true_classes]
y_pred_attentiveness = [map_to_attentiveness(idx) for idx in y_pred_classes]

# Calculate attentiveness metrics
attentiveness_cm = confusion_matrix(y_true_attentiveness, y_pred_attentiveness, labels=["Attentive", "Non-Attentive"])
attentiveness_cr = classification_report(y_true_attentiveness, y_pred_attentiveness)

# Plot attentiveness confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(attentiveness_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Attentive", "Non-Attentive"], 
            yticklabels=["Attentive", "Non-Attentive"])
plt.title('Attentiveness Confusion Matrix', fontweight='bold', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.tight_layout()
plt.savefig("attentiveness_confusion_matrix.png", dpi=300)
plt.show()

# Print attentiveness classification report
print("\nAttentiveness Classification Report:")
print(attentiveness_cr)

# Save final model
model.save("Face.keras")
print("Model saved as 'Face.keras'")