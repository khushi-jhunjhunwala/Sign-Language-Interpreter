import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize mediapipe hands module for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory path where the data is stored
DATA_DIR = './data'

data = []
labels = []

# Iterate through the directories in DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):  # Check if it's a directory
        # Iterate through the images in the directory
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            data_aux = []
            # Read the image using OpenCV
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            # Convert the image from BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect hands and their landmarks in the image
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                # Extract hand landmark coordinates
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)

                # Append the extracted landmarks and associated label to the dataset
                data.append(data_aux)
                labels.append(dir_)

# Serialize the data and labels and store them in a pickle file
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()