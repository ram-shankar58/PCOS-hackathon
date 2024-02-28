import numpy as np
import pandas as pd
import cv2
import os

def detect_yellow(base_image_path, image_path):

    fools=[]
    # Read the image
    for img in image_path:

        image = cv2.imread(os.path.join(base_image_path, img))

    # Convert BGR to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for yellow color
        lower_yellow = np.array([22, 50, 50])
        upper_yellow = np.array([60, 255, 255])

    # Create a mask for yellow color
        mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)


    # Apply the mask to the original image
        yellow_regions = np.any(mask)
        if(yellow_regions):
            fools.append(1)
        else:
            fools.append(0)

    yellow=np.array(fools)
    return yellow


impath='Dataset\images'
impath = os.listdir(impath)
cancer=detect_yellow('Dataset\images', impath)
df = pd.DataFrame({"Image": impath, "Prediction": cancer})
df.to_csv('predictions.csv', index=False)