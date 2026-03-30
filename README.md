# Fallguard-Edge-Demo
Minimal real-time working prototype for edge-based fall detection.

##  What is done so far
*  **Webcam Integration**: Successfully capturing live video feed using OpenCV.
*  **Pose Estimation**: Integrated Google MediaPipe's Pose Landmarker model to detect human keypoints in real-time.
*  **Fall Detection Logic**: Implemented the core logic to calculate angles between the shoulder, hip, and ankle to accurately detect a fall.
*  **Visual Feedback**: Real-time visual overlays on the webcam feed showing the pose skeleton and the current status (e.g., "Normal" vs "Fall Detected!").

##  What to do next
*  **Audio Alerts**: Implement a simple alert sound that triggers immediately when a fall is detected.
*  **Logic Refinement**: Tweak the angle thresholds to reduce false positives (e.g., bending down vs. falling).
*  **Edge Optimization**: Optimize the frame processing to ensure it runs smoothly on low-power edge devices.
*  **Robust Testing**: Test the system under different lighting conditions and camera angles.
