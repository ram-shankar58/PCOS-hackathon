import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import BinaryAccuracy
from densenetmodel import create_model
import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
import cv2

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