# %%

import os
import cv2
import numpy as np 
import pandas as pd

# %%
imagefolder='Dataset\PCOSGen-train\PCOSGen-train\images'
images=[]
labelslist=[]
size=(280,280)



# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator





# %%
labels=pd.read_excel('Dataset\PCOSGen-train\PCOSGen-train\class_label.xlsx')
labels['Healthy']=labels['Healthy'].astype(int)
numrows=len(labels)
labels.dropna(inplace=True)
numrowsdrop=numrows-len(labels)
X=labels['imagePath']
y=labels['Healthy']

healthy=labels[labels['Healthy']==1]['imagePath'].tolist()
unhealthy=labels[labels['Healthy']==0]['imagePath'].tolist()



# %%
output_folder_healthy = 'Dataset/healthy'
output_folder_unhealthy = 'Dataset/unhealthy'

# Ensure the output folders exist, or create them if not
os.makedirs(output_folder_healthy, exist_ok=True)
os.makedirs(output_folder_unhealthy, exist_ok=True)

# %%
def resize_and_save_images(image_paths, output_folder):
    for image_path in image_paths:
        img = cv2.imread(os.path.join(imagefolder, image_path))

        if img is not None:
            img_resized = cv2.resize(img, size)
            

            # Save resized image to the output folder
            output_path = os.path.join(output_folder, os.path.basename(image_path))
            cv2.imwrite(output_path, img_resized)
        else:
            print(f"Failed to read or resize image: {image_path}")

# %%
resize_and_save_images(healthy, output_folder_healthy)

# Resize and save unhealthy images
resize_and_save_images(unhealthy, output_folder_unhealthy)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train, random_state=42)


# %%
np.save('Dataset/X_train.npy', X_train)
np.save('Dataset/y_train.npy', y_train)
np.save('Dataset/X_test.npy', X_test)
np.save('Dataset/y_test.npy', y_test)



