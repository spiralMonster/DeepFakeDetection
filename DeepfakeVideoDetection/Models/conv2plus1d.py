import tensorflow as tf
from tensorflow.keras.layers import Layer,Conv3D


class Conv2Plus1D(Layer):
    def __init__(self, filters, kernel_size, activation, kernel_initializer, padding, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer

        # Initializing layers:
        self.ConvSpatial = Conv3D(
            filters=self.filters,
            kernel_size=(1, self.kernel_size[1], self.kernel_size[2]),
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            padding=self.padding
        )

        self.ConvDepth = Conv3D(
            filters=self.filters,
            kernel_size=(self.kernel_size[0], 1, 1),
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            padding=self.padding
        )

    def build(self, input_shape):
        self.ConvSpatial.build(input_shape)
        self.ConvDepth.build((input_shape[0], input_shape[1], input_shape[2], input_shape[3], self.filters))

    def call(self, inputs):
        x = self.ConvSpatial(inputs)
        x = self.ConvDepth(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3], self.filters)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer
        })
        return config

