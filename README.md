# Real-Time Pose-Based Attention Analysis on Raspberry Pi

## Overview

This project leverages a Raspberry Pi 4 B+ for real-time pose estimation in a classroom environment. It aims to analyze a student's posture using the MoveNet SinglePose Lightning model (a TensorFlow Lite model) to determine whether the student is paying attention. The system processes live video from a camera, extracts key body points (keypoints), applies heuristic analysis (such as head tilt, head position, and hand positions), and finally logs the data for future training or further analysis.

## Features

- **Real-Time Video Capture:** Uses OpenCV to stream live video from the camera.
- **Pose Estimation:** Utilizes a pre-trained TensorFlow Lite model (MoveNet SinglePose Lightning) to detect 17 keypoints (e.g., nose, eyes, shoulders, wrists).
- **Feature Extraction & Attention Analysis:**
  - **Head Tilt Calculation:** Determines the head angle based on the positions of the eyes.
  - **Posture Analysis:** Checks whether the nose position is above the shoulder level (to detect head drooping).
  - **Hand Position Check:** Identifies raised hands if a wrist is above its corresponding shoulder.
  - **Attention Status:** Combines these features to classify the frame as "Attentive" or "Not Attentive" with reasons.
- **Data Logging:** For every processed frame, logs the following to a CSV file:
  - Timestamp
  - Head tilt angle
  - Overall attention status
  - Feedback (e.g., "Head drooping", "Excessive head tilt", "Hand raised")
  - Raw keypoints data (JSON-encoded)


## Installation & Setup

### Requirements

- **Hardware:** Raspberry Pi 4 B+ (with camera module or USB webcam)
- **Operating System:** Raspberry Pi OS
- **Python Version:** Python 3.x

### Dependencies

Install the required Python libraries using `pip3`:
pip3 install opencv-python numpy tensorflow

### Clone the Repository
git clone https://github.com/Nandu-Kris7/Interest-Classification-using-object-Detection-on-Raspberry-Pi.git

## How It Works

### 1. Real-Time Pose Estimation

#### Video Capture
- **Description:**  
  The system uses OpenCV to capture frames from the connected camera.

#### Frame Preprocessing
- **Description:**  
  Each frame is resized to match the input dimensions of the MoveNet model (e.g., 192×192 pixels) and converted to `uint8` format, which is required for the quantized model.

#### Inference
- **Description:**  
  The TensorFlow Lite interpreter processes the preprocessed frame to obtain 17 keypoints representing body parts such as the eyes, nose, shoulders, and wrists.

---

### 2. Feature Extraction & Attention Analysis

#### Head Tilt
- **Description:**  
  Calculated by determining the angle between the left and right eyes. This helps assess if the head is tilted excessively.

#### Posture (Head Drooping)
- **Description:**  
  The vertical position of the nose is compared with the average y-coordinate of the shoulders. If the nose is too low relative to the shoulders, it may indicate head drooping.

#### Hand Position
- **Description:**  
  The system checks if the wrist keypoints are positioned higher than their corresponding shoulders, which can suggest the hand is raised—potentially a sign of distraction.

#### Attention Status
- **Description:**  
  Based on the above heuristics, the system flags frames where the student might be inattentive (e.g., excessive head tilt, head drooping, raised hand) and overlays the analysis result onto the video display.

---

### 3. Data Logging

#### CSV File Logging
- **Description:**  
  Each processed frame's data is recorded in a CSV file (`pose_data.csv`). Every row in the file includes:
  - **Timestamp:** When the frame was processed.
  - **Head Tilt:** The computed head tilt angle.
  - **Attention Status:** Boolean flag indicating if the student is deemed attentive.
  - **Feedback:** Descriptive feedback (e.g., "Head drooping") indicating potential issues.
  - **Keypoints:** Raw keypoints data (JSON-encoded) for potential use in model retraining or deeper analysis.

---

## Future Work & Enhancements

- **Model Fine-Tuning:**  
  You can use the collected CSV data (with labeled examples) to train a custom attention classifier, thereby improving the robustness of attention assessment.

- **Performance Optimization:**  
  Explore hardware accelerators like the Google Coral USB Accelerator or Intel Neural Compute Stick 2 to increase inference speed for continuous real-time applications.

- **Enhanced Feature Extraction:**  
  Integrate additional features such as body lean, eye tracking, and facial expressions for a more nuanced detection of student engagement.

- **User Interface Development:**  
  Create a user-friendly interface or dashboard for live monitoring and historical data visualization that could integrate with existing classroom management systems.

- **Integration with E-Learning Systems:**  
  Deploy the system in classroom environments to dynamically adjust teaching methods or provide real-time feedback to educators based on student engagement metrics.

---

## Conclusion

This project demonstrates a practical application of machine learning using embedded devices like the Raspberry Pi. With real-time video processing, pose estimation using TensorFlow Lite, and systematic data logging, the system provides foundational insights into student behavior. Future developments will further enhance the analysis capabilities and integration with educational systems, promoting interactive and adaptive learning experiences.

