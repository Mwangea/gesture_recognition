import cv2

def preprocess_frame(frame, target_size=(128, 128)):
    img = cv2.resize(frame, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img