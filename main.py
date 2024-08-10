import os
import argparse
from data_collection.collect_data import collect_gesture_data
from data_collection.gesture_definations import gestures
from model.train_model import train_and_save_model
from recognition.real_time_recognition import run_recognition

def main():
    parser = argparse.ArgumentParser(description="Gesture Recognition System")
    parser.add_argument('--collect', action='store_true', help='Collect gesture data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--recognize', action='store_true', help='Run real-time recognition')
    args = parser.parse_args()

    if args.collect:
        collect_gesture_data(gestures)
    elif args.train:
        train_and_save_model()
    elif args.recognize:
        run_recognition()
    else:
        print("Please specify an action: --collect, --train, or --recognize")

if __name__ == "__main__":
    main()