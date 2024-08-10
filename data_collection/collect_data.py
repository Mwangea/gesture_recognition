import cv2
import os
from .gesture_definations import gestures

def collect_gesture_data(gestures, num_samples=100):
    cap = cv2.VideoCapture(0)
    
    if not os.path.exists('gesture_data'):
        os.makedirs('gesture_data')
    
    for gesture_name, description in gestures.items():
        print(f"Collecting data for: {gesture_name}")
        print(f"Description: {description}")
        print("Press 'c' to capture an image, 'n' for next gesture, 'q' to quit")
        
        count = 0
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Count: {count}/{num_samples}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Collect Gesture Data', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                filename = f'gesture_data/{gesture_name}_{count}.jpg'
                cv2.imwrite(filename, frame)
                count += 1
                print(f'Saved {filename}')
            elif key == ord('n'):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_gesture_data(gestures)