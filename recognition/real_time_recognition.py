import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from utils.image_utils import preprocess_frame

def recognize_gesture(model, frame, class_names):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(np.expand_dims(processed_frame, axis=0))
    class_idx = np.argmax(prediction)
    return class_names[class_idx]

def run_recognition():
    model = tf.keras.models.load_model('gesture_model.h5')
    le = LabelEncoder()
    le.classes_ = np.load('label_encoder.npy', allow_pickle=True)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gesture = recognize_gesture(model, frame, le.classes_)
        
        cv2.putText(frame, f'Gesture: {gesture}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Gesture Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()