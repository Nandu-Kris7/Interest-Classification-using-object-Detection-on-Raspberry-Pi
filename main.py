import cv2
import numpy as np
import tensorflow as tf
import math
import csv
import time
import json

# Path to the MoveNet TFLite model (download from TensorFlow Hub)
MODEL_PATH = "/media/kris/NITRO/movenet_lightning.tflite"

# Load the TensorFlow Lite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Model input shape: [1, input_size, input_size, 3]
input_shape = input_details[0]['shape']
input_size = input_shape[1]  # e.g., 192 for MoveNet Lightning

def preprocess_frame(frame):
    """
    Resize the frame to the required input dimensions and cast to uint8.
    This is appropriate when using a quantized TFLite model.
    """
    img = cv2.resize(frame, (input_size, input_size))
    input_data = np.expand_dims(img, axis=0).astype(np.uint8)
    return input_data

def calculate_head_tilt(keypoints, conf_threshold=0.4):
    """
    Calculate head tilt (in degrees) using the positions of the eyes.
    keypoints: an array of shape [17, 3] where each is [y, x, score].
    Uses keypoints: 1: Left eye, 2: Right eye.
    """
    nose = keypoints[0]
    left_eye = keypoints[1]
    right_eye = keypoints[2]
    
    # Check that the keypoints meet the confidence threshold.
    if nose[2] > conf_threshold and left_eye[2] > conf_threshold and right_eye[2] > conf_threshold:
        # Calculate angle between the eyes.
        dx = right_eye[1] - left_eye[1]
        dy = right_eye[0] - left_eye[0]
        angle = math.degrees(math.atan2(dy, dx))
        return angle
    return 0.0

def analyze_attention(keypoints, head_tilt, conf_threshold=0.4):
    """
    Analyze the pose to infer attention.
    
    Heuristics used:
      - **Head Position:** Average y-coordinate of shoulders (points 5 and 6) versus nose (point 0).
        If the nose is not sufficiently above the shoulders (by a small margin), flag as "head drooping."
      - **Head Tilt:** If the absolute head tilt angle exceeds 15 degrees, flag as "excessive head tilt."
      - **Hand Positions:** Check if the left or right wrist (points 9 and 10) is above the corresponding shoulder.
    
    Returns a tuple (attention, feedback) where:
      - `attention` is a boolean (True if the person is considered attentive).
      - `feedback` is a list of strings that explain factors suggesting inattention.
    """
    attention = True
    feedback = []
    
    # Extract keypoints.
    nose = keypoints[0]
    left_eye = keypoints[1]
    right_eye = keypoints[2]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    
    # Compute the average shoulder y-coordinate.
    avg_shoulder_y = (left_shoulder[0] + right_shoulder[0]) / 2.0

    # Head Position: Nose should be higher (lower y) than shoulders.
    if nose[0] > avg_shoulder_y - 0.05:
        attention = False
        feedback.append("Head drooping")
    
    # Head Tilt: Check if tilt exceeds 15 degrees.
    if abs(head_tilt) > 15:
        attention = False
        feedback.append("Excessive head tilt")
        
    # Hand Positions: Check if either wrist is raised above its corresponding shoulder.
    if left_wrist[2] > conf_threshold and left_wrist[0] < left_shoulder[0]:
        attention = False
        feedback.append("Left hand raised")
    if right_wrist[2] > conf_threshold and right_wrist[0] < right_shoulder[0]:
        attention = False
        feedback.append("Right hand raised")
    
    return attention, feedback

# Open a CSV file to store data.
csv_filename = "/home/kris/Documents/Project/pose_data.csv"
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)
# Write header: Timestamp, Head Tilt, Attention, Feedback, and Keypoints (as a JSON string).
csv_writer.writerow(["timestamp", "head_tilt", "attention", "feedback", "keypoints"])

# Open video capture from the default camera.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame and run inference.
    input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # The model output: [1, 1, 17, 3] -> one detection with 17 keypoints.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    keypoints = keypoints_with_scores[0, 0, :, :]  # Shape: (17, 3)

    # Draw keypoints on the original frame.
    h, w, _ = frame.shape
    for kp in keypoints:
        if kp[2] > 0.4:  # Only plot confident keypoints.
            cx = int(kp[1] * w)
            cy = int(kp[0] * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # Compute head tilt.
    head_tilt = calculate_head_tilt(keypoints)
    
    # Analyze attention.
    attention, feedback = analyze_attention(keypoints, head_tilt)
    
    # Prepare overlay text and color based on attention analysis.
    if attention:
        status_text = "Attentive"
        status_color = (0, 255, 0)  # Green
    else:
        status_text = "Not Attentive: " + ", ".join(feedback)
        status_color = (0, 0, 255)  # Red
        
    # Display the head tilt and attention status on the frame.
    cv2.putText(frame, f"Head Tilt: {head_tilt:.1f} deg", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, status_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    # Display the annotated video feed.
    cv2.imshow("Pose Detection & Attention Analysis", frame)
    
    # --- Data Logging ---
    # Get a timestamp for the current frame.
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    # Convert feedback list and keypoints to JSON strings for storage.
    feedback_json = json.dumps(feedback)
    # Convert keypoints to list and then to JSON.
    keypoints_json = json.dumps(keypoints.tolist())
    
    # Write the data row.
    csv_writer.writerow([timestamp, head_tilt, attention, feedback_json, keypoints_json])
    csv_file.flush()  # Flush to ensure data is written to disk
    
    # Break the loop if 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up: Close the video stream and the CSV file.
cap.release()
cv2.destroyAllWindows()
csv_file.close()
