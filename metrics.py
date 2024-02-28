from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from densenetmodel import create_model
from keras.metrics import BinaryAccuracy
import cv2
import numpy as np
import os

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1

# Assuming y_true and y_pred are the true and predicted labels, respectively
def load_images(image_paths, image_size=(280, 280)):
    """
    This function loads grayscale images from disk based on their file paths and resizes them to a consistent size.
    """
    images = []

    for img_file in image_paths:
        # create the full input path and read the file
        image_path = os.path.join('Dataset\PCOSGen-train\PCOSGen-train\images', img_file)
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image =cv2.resize(image, image_size)
        # duplicate the grayscale image across three channels
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        images.append(image)

    return np.array(images)

def test_model(x_test, y_test):
    # dimensions of our images.
    img_width, img_height = 280, 280

    # specify the number of classes
    num_classes = 2

    # create the base pre-trained model
    model = create_model((img_width, img_height, 3), num_classes)

    # load the saved weights
    model.load_weights('model_weights.h5')

    # predict the output 
    predictions = model.predict(x_test)

    # get the class with highest probability for each sample
    y_pred = np.argmax(predictions, axis=1)

    # calculate accuracy
    accuracy = BinaryAccuracy()
    accuracy.update_state(y_test, y_pred)
    test_accuracy = accuracy.result().numpy()

    return y_pred, test_accuracy


def main():
    # load data
    x_test_paths = np.load('Dataset/X_test.npy', allow_pickle=True)
    y_test = np.load('Dataset/y_test.npy', allow_pickle=True)

    x_test = load_images(x_test_paths)

    # test model
    y_pred, test_accuracy = test_model(x_test, y_test)

    # print evaluation metrics
    print('Test accuracy:', test_accuracy)

if __name__ == "__main__":
    main()
