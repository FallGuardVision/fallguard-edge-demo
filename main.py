import cv2
import mediapipe as mp

def main():
    print("Initializing FallGuard Edge Camera Module with MediaPipe...")
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Configure 1080p @ 30fps
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera initialized. Press 'ESC' or 'q' to exit.")
    
    frame_count: int = 0
    landmark_names = [lm.name for lm in mp_pose.PoseLandmark]

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Optimize pipeline
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect 33 keypoints
            results = pose.process(image_rgb)

            # Draw the pose annotations on the image
            image.flags.writeable = True
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

                # Print coordinates every 30 frames to terminal
                if frame_count % 30 == 0:
                    print(f"\n--- Frame {frame_count} | 33 Keypoint Coordinates ---")
                    for i, lm in enumerate(results.pose_landmarks.landmark):
                        print(f"  [{i:02d}] {landmark_names[i]:<25} x={lm.x:.4f}  y={lm.y:.4f}  z={lm.z:.4f}  vis={lm.visibility:.2f}")
                
            cv2.imshow('FallGuard Edge - Layer 1 & 2 Demo (33 Keypoints)', image)
            frame_count += 1
            
            key = cv2.waitKey(5) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
