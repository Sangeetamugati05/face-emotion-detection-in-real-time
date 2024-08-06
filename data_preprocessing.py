import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer

def load_data(data_dir):
    data = []
    labels = []
    for emotion in os.listdir(data_dir):
        emotion_dir = os.path.join(data_dir, emotion)
        if not os.path.isdir(emotion_dir):
            continue
        for image_name in os.listdir(emotion_dir):
            image_path = os.path.join(emotion_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (48, 48))
            data.append(image)
            labels.append(emotion)
    return np.array(data), np.array(labels)

def preprocess_input(x):
    x = x.astype('float32')
    x = x / 255.0
    x = x - 0.5
    x = x * 2.0
    return x

def load_and_preprocess_data(train_dir, test_dir):
    x_train, y_train = load_data(train_dir)
    x_test, y_test = load_data(test_dir)
    
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)
    
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)
    
    return x_train, y_train, x_test, y_test, lb.classes_