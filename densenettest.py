import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import BinaryAccuracy
from densenetmodel import create_model
import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
import cv2

import matplotlib.pyplot as plt
import random

def plot_evaluation_metrics(history):
    # Plot accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_interpretability(images, predictions, class_names):
    # Select 5 random images
    random_indices = random.sample(range(len(images)), 5)
    selected_images = images[random_indices]
    selected_predictions = predictions[random_indices]

    # Plot the images and their predicted classes
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, (image, prediction) in enumerate(zip(selected_images, selected_predictions)):
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(f'Predicted Class: {class_names[prediction]}')
    plt.show()

def load_images(image_paths, image_size=(280, 280)):
    """
    This function loads grayscale images from disk based on their file paths and resizes them to a consistent size.
    """
    images = []

    for img_file in image_paths:
        # create the full input path and read the file
        image_path = os.path.join('Dataset\PCOSGen-train\PCOSGen-train\images', img_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        cv2.resize(image, (280,280))

        # duplicate the grayscale image across three channels
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        images.append(image)

    return np.array(images)


def test_model(x_train, y_train, x_test, y_test):
    # dimensions of our images.
    img_width, img_height = 280, 280

    # specify the number of classes
    num_classes = 2

    # create the base pre-trained model
    # create the base pre-trained model
    model = create_model((img_width, img_height, 3), num_classes)


    # define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

    # fit the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping, learning_rate_reduction])

    # save weights to file
    model.save_weights('model_weights.h5')

    # predict the output 
    predictions = model.predict(x_test)

    # get the class with highest probability for each sample
    y_pred = np.argmax(predictions, axis=1)

    return y_pred

from keras.utils import to_categorical

def main():
    # load data
    x_train_paths = np.load('Dataset/X_train.npy', allow_pickle=True)
    y_train = np.load('Dataset/y_train.npy', allow_pickle=True)
    x_test_paths = np.load('Dataset/X_test.npy', allow_pickle=True)
    y_test = np.load('Dataset/y_test.npy', allow_pickle=True)

    x_train = load_images(x_train_paths)
    x_test = load_images(x_test_paths)

    # test model
    y_pred = test_model(x_train, y_train, x_test, y_test)

    # calculate accuracy
    accuracy = BinaryAccuracy()
    accuracy.update_state(y_test, y_pred)
    print('Test accuracy:', accuracy.result().numpy())

    class_names = [0, 1]  # Replace with your class names
    plot_interpretability(x_test, y_pred, class_names)

if __name__ == "__main__":
    main()

