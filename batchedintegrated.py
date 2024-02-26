from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os 
import cv2
import numpy as np
from votingclassifier import *

def load_images(image_paths, image_size=(280, 280)):
    """
    This function loads grayscale images from disk based on their file paths and resizes them to a consistent size.
    """
    images = []

    for img_file in image_paths:
        # create the full input path and read the file
        image_path = os.path.join('Dataset/images/', img_file)
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # duplicate the grayscale image across three channels
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        images.append(image)

    return np.array(images)
# Assuming X_train and y_train are already defined
x_train_paths = np.load('Dataset/X_train.npy', allow_pickle=True)
y_train = np.load('Dataset/y_train.npy', allow_pickle=True)
x_test_paths = np.load('Dataset/X_test.npy', allow_pickle=True)
y_test = np.load('Dataset/y_test.npy', allow_pickle=True)

x_train = load_images(x_train_paths)
x_test = load_images(x_test_paths)
# Split the data into training and validation sets
#X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Create the complex model
complex_model = create_complex_model(input_shape=x_train.shape[1:])

#remove argumentss for create complex model if you want later



# Train the complex model
complex_model.fit(x_train[:10], y_train[:10])

# Validate the model on the first 10 images
y_pred = complex_model.predict(x_test[:10])
accuracy = accuracy_score(y_test[:10], y_pred)
print("Validation Accuracy for first 10 images:", accuracy)
