from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
from densenetmodel import create_model
from keras.metrics import BinaryAccuracy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_datagen():
    datagen = ImageDataGenerator()
    return datagen

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

def test_model(x_train, y_train, x_test, y_test):
    # dimensions of our images.
    img_width, img_height = 280, 280

    # specify the number of classes
    num_classes = 2

    # create the base pre-trained model
    model = create_model((img_width, img_height, 3), num_classes)

    # define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

    # create data generator
    datagen = create_datagen()

    # fit the model
    model.fit(datagen.flow(x_train, y_train, batch_size=32), validation_data=(x_test, y_test), epochs=100, callbacks=[early_stopping, learning_rate_reduction])

    # save weights to file
    model.save_weights('model_weights.h5')

    # predict the output 
    predictions = model.predict(x_test[:10])

    # get the class with highest probability for each sample
    y_pred = np.argmax(predictions, axis=1)

    return y_pred

def main():
    # load data
    x_train_paths = np.load('Dataset/X_train.npy', allow_pickle=True)
    y_train = np.load('Dataset/y_train.npy', allow_pickle=True)
    x_test_paths = np.load('Dataset/X_test.npy', allow_pickle=True)
    y_test = np.load('Dataset/y_test.npy', allow_pickle=True)

    x_train = load_images(x_train_paths)
    x_test = load_images(x_test_paths)

    # test model
    y_pred = test_model(x_train[:10], y_train[:10], x_test[:10], y_test[:10])

    # calculate accuracy
    accuracy = BinaryAccuracy()
    accuracy.update_state(y_test[:10], y_pred[:10])
    print('Test accuracy:', accuracy.result().numpy())

if __name__ == "__main__":
    main()
