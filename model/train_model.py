import numpy as np
from .model_architecture import create_model
from data_processing.preprocess_data import prepare_data

def train_and_save_model():
    X_train, X_test, y_train, y_test, le = prepare_data()
    
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))

    model = create_model(input_shape, num_classes)
    model.fit(X_train, y_train, epochs=50, validation_split=0.2)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc}')

    model.save('gesture_model.h5')
    np.save('label_encoder.npy', le.classes_)

if __name__ == "__main__":
    train_and_save_model()