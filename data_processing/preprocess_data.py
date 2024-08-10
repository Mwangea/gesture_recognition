import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data_dir='gesture_data', img_size=(128, 128)):
    images = []
    labels = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(data_dir, filename))
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0  # Normalize
            
            images.append(img)
            labels.append(filename.split('_')[0])
    
    return np.array(images), np.array(labels)

def prepare_data():
    X, y = preprocess_data()
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, le

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le = prepare_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(le.classes_)}")