from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
import time

# load models
model_face = load_model('Face.keras')
model_eyes = load_model('Eyes.keras')

# open webcam
webcam = cv2.VideoCapture(0)

# Define classes for both models
face_classes = ["Forward", "Left Mirror", "Radio", "Rearview", "Speedometer", "Lap"]
eyes_classes = ["Closed", "Opened"]

# Attentiveness mapping for face model
ATTENTIVE_CLASSES = ["Forward", "Rearview", "Radio", "Speedometer"]
NON_ATTENTIVE_CLASSES = ["Left Mirror", "Lap"]

# Define attentiveness multipliers for each class
# Higher values indicate more contribution to non-attentive state
face_multipliers = {
    "Forward": 0.1,      # Very attentive - looking forward
    "Left Mirror": 1.5,  # Not attentive - looking away
    "Radio": 0.4,        # Somewhat attentive - brief distraction
    "Rearview": 0.3,     # Somewhat attentive - checking surroundings
    "Speedometer": 0.3,  # Somewhat attentive - brief glance down
    "Lap": 1.2           # Not attentive - looking down
}

eyes_multipliers = {
    "Closed": 1.0,       # Not attentive - eyes closed
    "Opened": 0.2        # Attentive - eyes open
}

# Threshold for determining attentiveness (adjust as needed)
ATTENTIVENESS_THRESHOLD = 0.8

# Initialize variables for timed processing
last_process_time = time.time()
process_interval = 1.0  # Process models every 1 second
face_text = ""
eyes_text = ""
attentiveness_text = ""
attentiveness_score = 0

# Initialize attentiveness_color with a default value
attentiveness_color = (255, 255, 255)  # Default to white

# loop through frames
while webcam.isOpened():
    # read frame from webcam 
    status, frame = webcam.read()
    
    if not status:
        break

    # apply face detection
    face_regions, confidence = cv.detect_face(frame)

    # loop through detected faces
    for idx, f in enumerate(face_regions):
        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # Check if it's time to process with the models (every 1 second)
        current_time = time.time()
        if current_time - last_process_time >= process_interval:
            # Reset timer
            last_process_time = current_time
            
            # Preprocessing for face model
            face_resized = cv2.resize(face_crop, (224, 224))  # EfficientNet size
            face_normalized = face_resized.astype("float") / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            # Face model prediction
            face_conf = model_face.predict(face_batch, verbose=0)[0]
            face_idx = np.argmax(face_conf)
            face_label = face_classes[face_idx]
            face_text = f"Face: {face_label} ({face_conf[face_idx]:.2f})"
            
            # Extract eye region (upper half of face is a simple approximation)
            face_height = face_crop.shape[0]
            eyes_crop = face_crop[0:int(face_height/2), :]
            
            # Preprocessing for eyes model
            eyes_resized = cv2.resize(eyes_crop, (224, 224))  # EfficientNet size
            eyes_normalized = eyes_resized.astype("float") / 255.0
            eyes_batch = np.expand_dims(eyes_normalized, axis=0)
            
            # Eyes model prediction
            eyes_conf = model_eyes.predict(eyes_batch, verbose=0)[0]
            eyes_idx = np.argmax(eyes_conf)
            eyes_label = eyes_classes[eyes_idx]
            eyes_text = f"Eyes: {eyes_label} ({eyes_conf[eyes_idx]:.2f})"
            
            # Calculate attentiveness score - weighted combination of face and eye predictions
            # Higher score means less attentive
            attentiveness_score = 0
            
            # Add face model contribution
            for i, cls in enumerate(face_classes):
                if cls in face_multipliers:
                    attentiveness_score += face_conf[i] * face_multipliers[cls]
            
            # Add eyes model contribution
            for i, cls in enumerate(eyes_classes):
                if cls in eyes_multipliers:
                    attentiveness_score += eyes_conf[i] * eyes_multipliers[cls]
            
            # Determine attentiveness state
            if attentiveness_score > ATTENTIVENESS_THRESHOLD:
                attentiveness_text = f"NON-ATTENTIVE ({attentiveness_score:.2f})"
                attentiveness_color = (0, 0, 255)  # Red for non-attentive
            else:
                attentiveness_text = f"ATTENTIVE ({attentiveness_score:.2f})"
                attentiveness_color = (0, 255, 0)  # Green for attentive
                
        # Position text for labels
        Y1 = startY - 40 if startY - 40 > 10 else startY + 10
        Y2 = Y1 + 30  # Position second label below first label
        Y3 = Y2 + 30  # Position third label below second label

        # Display all labels
        cv2.putText(frame, face_text, (startX, Y1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)  # Blue for face
        cv2.putText(frame, eyes_text, (startX, Y2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 165, 0), 2)  # Orange for eyes
        cv2.putText(frame, attentiveness_text, (startX, Y3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, attentiveness_color, 2)  # Color varies by attention state

    # display output
    cv2.imshow("Driver Attention Monitoring", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()