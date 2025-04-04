import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,BatchNormalization,MaxPooling2D,Dense,Flatten,Dropout
from .inception_layer import Inception_cell
from .skip_connection_layer import SkipConnection

class DeepFakeDetectionModel:
    def build_model(self,input_shape):
        inp=Input(shape=input_shape,dtype=tf.float32)
        x=Inception_cell(num_filters=16)(inp)
        x=BatchNormalization()(x)
        x=MaxPooling2D((2,2),padding='same')(x)

        x=Inception_cell(num_filters=32)(x)
        x=BatchNormalization()(x)
        x=MaxPooling2D((2,2),padding='same')(x)

        x=SkipConnection(num_filters=64,kernel_size=(3,3))(x)
        x=MaxPooling2D((2,2),padding='same')(x)

        x=SkipConnection(num_filters=128,kernel_size=(5,5))(x)
        x=MaxPooling2D((2,2),padding='same')(x)

        x=Flatten()(x)
        x=Dense(1024,activation='relu',kernel_initializer='he_uniform')(x)
        x=Dense(256,activation='relu',kernel_initializer='he_uniform')(x)
        x=Dropout(0.2)(x)

        x=Dense(128,activation='relu',kernel_initializer='he_uniform')(x)
        x=Dense(64,activation='relu',kernel_initializer='he_uniform')(x)

        x=Dense(16,activation='relu',kernel_initializer='he_uniform')(x)
        x=Dense(4,activation='relu',kernel_initializer='he_uniform')(x)
        x=Dense(2,activation='softmax',kernel_initializer='glorot_uniform')(x)

        model=Model(inputs=inp,outputs=x)
        return model












