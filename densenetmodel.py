from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

def create_model(input_shape, num_classes):
    # load DenseNet121 model without classification layers
    base_model = DenseNet121(include_top=False, input_shape=input_shape)

    # add global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    # and a logistic layer
    predictions = Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # compile the model
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    return model
