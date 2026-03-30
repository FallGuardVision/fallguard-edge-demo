import cv2
import mediapipe as mp
import time
import winsound

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
    last_alert_time: float = 0.0
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

                # --- FALL DETECTION LOGIC ---
                landmarks = results.pose_landmarks.landmark
                
                # Get relevant nodes
                ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                
                # Y is normalized [0.0, 1.0], 0 is top, 1 is bottom
                shoulder_y = (ls.y + rs.y) / 2
                hip_y = (lh.y + rh.y) / 2
                
                # Fetch visible landmarks to compute aspect ratio
                visible_lms = [lm for lm in landmarks if lm.visibility > 0.5]
                if len(visible_lms) > 5:
                    min_x = min([lm.x for lm in visible_lms])
                    max_x = max([lm.x for lm in visible_lms])
                    min_y = min([lm.y for lm in visible_lms])
                    max_y = max([lm.y for lm in visible_lms])
                    
                    bbox_w = max_x - min_x
                    bbox_h = max_y - min_y
                    aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0
                    
                    # Condition 1: Aspect ratio > 1 generally means laying horizontally
                    # Condition 2: Shoulders are lower than hips means upside-down / fallen over
                    if aspect_ratio > 1.2 or shoulder_y > hip_y:
                        cv2.putText(image, "FALL DETECTED!!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6, cv2.LINE_AA)
                        cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 255), 15)
                        
                        current_time = time.time()
                        if current_time - last_alert_time > 1.5:
                            print(f"\n[ALERT] Fall Detected at Frame {frame_count}! Aspect Ratio: {aspect_ratio:.2f}")
                            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                            last_alert_time = current_time

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
