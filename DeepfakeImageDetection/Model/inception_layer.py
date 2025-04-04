import tensorflow as tf
from tensorflow.keras.layers import Layer,Conv2D,Add

class Inception_cell(Layer):
    def __init__(self,num_filters,activation='relu',padding='same',**kwargs):
        super().__init__(**kwargs)
        self.num_filters=num_filters
        self.activation=activation
        self.padding=padding
        self.conv1=Conv2D(self.num_filters,(3,3),activation=self.activation,padding=self.padding)
        self.conv2=Conv2D(self.num_filters,(5,5),activation=self.activation,padding=self.padding)
        self.conv3=Conv2D(self.num_filters,(7,7),activation=self.activation,padding=self.padding)

    def call(self,input):
        x1=self.conv1(input)
        x2=self.conv2(input)
        x3=self.conv3(input)
        out=Add()([x1,x2,x3])
        return out

    def get_config(self):
        config=super().get_config()
        config.update({
            'num_filters':self.num_filters,
            'activation':self.activation,
            'padding':self.padding
        })
        return config


