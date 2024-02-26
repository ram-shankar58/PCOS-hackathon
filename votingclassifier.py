# Import necessary libraries
from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input, UpSampling2D, concatenate
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import ViTModel, ViTConfig
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam,SGD
from keras.applications.densenet import DenseNet121
from keras.layers import GlobalAveragePooling2D
from torch import nn
import torch 
from keras.applications.resnet50 import ResNet50

#Jai Ganesh! Hopefully this DBN works out

#for the RBM, k is Gibbs sampling steps to perform
#w, b, c are weights, biases of RBM learned during training
#forward performs k steps of Gbibbs sampling returns final state of visuible units

#Use Gaussian Bernoulli RBM for contiunuous data if this shit doesnt work smh

class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units, k=2):
        super(RBM, self).__init__()
        self.visible_units=visible_units
        self.hidden_units=hidden_units
        self.k=k
        self.W=nn.Parameter(torch.randn(visible_units, hidden_units)*0.01)
        self.b=nn.Parameter(torch.zeros(visible_units))
        self.c=nn.Parameter(torch.zeros(hidden_units))

    def forward(self, v):
        for i in range(self.k):
            p_h_given_v = torch.sigmoid(torch.matmul(v, self.W) + self.c)
            h = torch.bernoulli(p_h_given_v)
            p_v_given_h = torch.sigmoid(torch.matmul(h, self.W.t()) + self.b)
            v = torch.bernoulli(p_v_given_h)
        return v
    

def convert_data_to_dbn_format(X):
    # Assuming X is a numpy array of images with shape (num_images, height, width)
    # Flatten each image into a 1D array
    X_dbn = X.reshape(X.shape[0], -1)
    return X_dbn

def convert_labels_to_dbn_format(y):
    # Assuming y is a numpy array of labels with shape (num_images,)
    # DBN expects labels as float32
    y_dbn = y.astype('float32')
    return y_dbn

def convert_dbn_predictions_to_original_format(y_dbn):
    # Assuming y_dbn is a numpy array of DBN predictions with shape (num_images,)
    # Convert predictions to binary labels (0 or 1)
    y = (y_dbn > 0.5).astype('int')
    return y

            
#Wrapper class for integrating the DBN with the ensembling model
#im questioning my existence ;-\
class DBNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, dbn):
        self.dbn = dbn

    def fit(self, X, y):
        # Convert data to the format expected by the DBN
        X_dbn = convert_data_to_dbn_format(X)
        y_dbn = convert_labels_to_dbn_format(y)

        # Train the DBN
        self.dbn.fit(X_dbn, y_dbn)

    def predict(self, X):
        # Convert data to the format expected by the DBN
        X_dbn = convert_data_to_dbn_format(X)

        # Use the DBN to make predictions
        y_dbn = self.dbn.predict(X_dbn)

        # Convert the DBN's predictions to the original label format
        y = convert_dbn_predictions_to_original_format(y_dbn)

        return y


class DBN(nn.Module):
    def __init__(self):
        super(DBN,self).__init__()
        self.rbm1=RBM(visible_units=28*28, hidden_units=500)
        self.rbm2=RBM(visible_units=500,hidden_units=200)
        self.rbm3=RBM(visible_units=200, hidden_units=50)
        self.classifier=nn.Linear(50,10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        h1 = self.rbm1(x)
        h2 = self.rbm2(h1)
        h3 = self.rbm3(h2)
        out = self.classifier(h3)
        return out

def unet(input_shape):
    inputs=Input(input_shape)
    conv1=Conv2D(64,3,activation='relu',padding='same')(inputs)
    conv1=Conv2D(64,3,activation='relu',padding='same')(conv1)
    pool1=MaxPooling2D(pool_size=(2,2))(conv1)
    
    conv2=Conv2D(64,3,activation='relu',padding='same')(pool1)
    conv2=Conv2D(64,3,activation='relu', padding='same')(conv2)
    pool2=MaxPooling2D(pool_size=(2,2))(conv2)

    conv3=Conv2D(64,3,activation='relu',padding='same')(pool2)
    conv3=Conv2D(64,3,activation='relu', padding='same')(conv3)
    pool3=MaxPooling2D(pool_size=(2,2))(conv3)

    conv4=Conv2D(64,3,activation='relu',padding='same')(pool3)
    conv4=Conv2D(64,3,activation='relu', padding='same')(conv4)
    pool4=MaxPooling2D(pool_size=(2,2))(conv4)

    conv5=Conv2D(64,3,activation='relu',padding='same')(pool4)
    conv5=Conv2D(64,3,activation='relu', padding='same')(conv5)
    pool5=MaxPooling2D(pool_size=(2,2))(conv5)

    conv6=Conv2D(64,3,activation='relu',padding='same')(pool5)
    conv6=Conv2D(64,3,activation='relu', padding='same')(conv5)
    pool6=MaxPooling2D(pool_size=(2,2))(conv5)

    conv7=Conv2D(64,3,activation='relu',padding='same')(pool6)
    conv7=Conv2D(64,3,activation='relu', padding='same')(conv7)

    #decoding layers starting bleow
    # Start of the expansive path
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv6], axis=-1)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(up6)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv8)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv5], axis=-1)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv4], axis=-1)
    conv10 = Conv2D(64, 3, activation='relu', padding='same')(up8)
    conv10 = Conv2D(64, 3, activation='relu', padding='same')(conv10)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv3], axis=-1)
    conv11 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv11 = Conv2D(64, 3, activation='relu', padding='same')(conv11)

    up10 = concatenate([UpSampling2D(size=(2, 2))(conv11), conv2], axis=-1)
    conv12 = Conv2D(64, 3, activation='relu', padding='same')(up10)
    conv12 = Conv2D(64, 3, activation='relu', padding='same')(conv12)

    up11 = concatenate([UpSampling2D(size=(2, 2))(conv12), conv1], axis=-1)
    conv13 = Conv2D(64, 3, activation='relu', padding='same')(up11)
    conv13 = Conv2D(64, 3, activation='relu', padding='same')(conv13)

# Final layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv13)
    return Model(inputs=[inputs],outputs=[outputs])
    

def create_complex_model(input_shape):
    # Define the DNN model
    dnn_model = Sequential()
    dnn_model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
    dnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    dnn_model.add(Dropout(0.25))
    dnn_model.add(Conv2D(128, kernel_size=3, activation='relu'))
    dnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    dnn_model.add(Dropout(0.25))
    dnn_model.add(Flatten())
    dnn_model.add(Dense(256, activation='relu'))
    dnn_model.add(Dropout(0.5))
    dnn_model.add(Dense(10, activation='softmax'))

    # DenseNet model
    densebase=DenseNet121(include_top=False, input_shape=input_shape)
    x=densebase.output
    x=GlobalAveragePooling2D()(x)

    x=Dense(1024, activation='relu')(x)

    predictions=Dense(1,activation='sigmoid')(x)
    densenetmodel=Model(inputs=densebase.input,outputs=predictions)

    # Define the machine learning models
    
    rf_model = RandomForestClassifier()
    svm_model = SVC(probability=True)
    gb_model = GradientBoostingClassifier()

#VIT model
    config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
    vit_model = ViTModel(config)
    input_layer = Input(shape=(224, 224, 3))

    #RESNET

    resnet_model=ResNet50(weights='imagnet', include_Top=False, input_shape=input_shape)
    x1=resnet_model.output
    x1=GlobalAveragePooling2D()(x)
    pred=Dense(1,activatoin='sigmoid')(x)
    resnet_model=Model(inputs=resnet_model.input, outputs=predictions)


# Connect the new input layer to the ViT model
    output_layer = vit_model(input_layer)

# Create a new model with the new input layer
    vit_model = Model(inputs=input_layer, outputs=output_layer)

#unet model
    unetmodel=unet(input_shape)


    #DBN MODEL
    dbn=DBN()

    dbn_Wrapper=DBNWrapper(dbn)
    

    #Any problem, remove dbn model

    # Define the complex model
    complex_model = VotingClassifier(estimators=[('dnn', dnn_model),('DBN',dbn_Wrapper), ('vit', vit_model),('resnet',resnet_model), ('rf', rf_model),('unet',unetmodel), ('svm', svm_model), ('gb', gb_model),('densenet',densenetmodel)], voting='soft')
    Momentum=SGD(lr=0.01, momentum=0.9)

    #If too slow use ADAM itself
    complex_model.compile(optimizer=Momentum, loss='binary_crossentropy', metrics=['accuracy'])
