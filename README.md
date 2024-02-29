## Project Report

## Abstract

Polycystic Ovary Syndrome (PCOS) is a prevalent endocrine disorder affecting women of reproductive age, characterized by symptoms such as irregular menstrual cycles, excessive androgen production, and polycystic ovaries. Early and accurate diagnosis is essential for effective management and treatment. This report presents a deep learning model based on the DenseNet121 architecture, tailored for the binary classification of PCOS from MRI images. The model leverages transfer learning, utilizing a pre-trained DenseNet121 network, and introduces additional fully-connected layers and a logistic layer with a sigmoid activation function for the final prediction. Compiled with the Adam optimizer and binary cross-entropy loss, the model aims to provide a reliable diagnostic tool for medical practitioners.

## Introduction

The integration of deep learning in medical diagnostics has shown promising results, particularly in image classification and detection tasks. With the challenge of manual PCOM detection from ultrasound images, deep learning models offer a powerful alternative for automatic analysis, potentially reducing the need for manual examination and the associated errors. This research focuses on the application of the DenseNet121 architecture, a convolutional neural network known for its efficiency in feature reuse and information flow, to detect PCOS from MRI images.

## Materials and Methods

### Dataset and Preprocessing

The dataset comprises MRI images of ovaries, which are used to train the deep learning model. Preprocessing steps include resizing and data augmentation to address the limited size of the dataset and improve the model's generalization capabilities.

### Model Architecture

The DenseNet121 model is employed as the base architecture, chosen for its dense connectivity pattern that ensures maximum information flow between layers in the network. The model is initialized with weights pre-trained on the ImageNet dataset, with the top classification layers excluded to allow for customization to the PCOS detection task.

Following the base model, a global average pooling layer is introduced to condense the feature maps into a single vector, reducing the number of parameters and preventing overfitting. Subsequently, a series of fully-connected layers with 1024, 512, 256, and 128 neurons are added, all utilizing the ReLU activation function to introduce non-linearity and aid in learning complex patterns in the data.

The final layer is a logistic layer with a sigmoid activation function, specifically designed for binary classification. This layer outputs a probability score indicating the likelihood of PCOS presence in the MRI image.

### Model Compilation

The model is compiled using the Adam optimizer, a popular choice for deep learning tasks due to its adaptive learning rate capabilities. The learning rate is set to 0.0001 to ensure gradual and stable convergence during training. Binary cross-entropy is selected as the loss function, suitable for binary classification problems, and accuracy is used as the metric to evaluate the model's performance.

## Results

The model for PCOS detection from MRI images is a novel application of deep learning in the medical field. This model has an accuracy of 75.6%, while testing on the split test data.
Thus, This documentation report outlines the development of a DenseNet121-based deep learning model for the detection of PCOS from MRI images. The model's architecture, which includes a pre-trained DenseNet121 base, additional fully-connected layers, and a logistic layer with a sigmoid activation function, is compiled with the Adam optimizer and binary cross-entropy loss. The model holds promise as a diagnostic tool for PCOS, with the potential to improve accuracy and efficiency in medical diagnostics.
