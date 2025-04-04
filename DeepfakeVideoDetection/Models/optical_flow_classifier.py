import tensorflow as tf
from tensorflow.keras.layers import Layer,Conv2D,Concatenate

class OpticalFlowClassifier(Layer):
    def __init__(self, layer_config, temporal_dim, **kwargs):
        super().__init__(**kwargs)
        self.layer_config = layer_config
        self.temporal_dim = temporal_dim

        # Initializing Intermediate layers:
        self.conv_layers = []
        for _ in range(self.temporal_dim):
            layer = Conv2D(
                filters=self.layer_config["filters"],
                kernel_size=self.layer_config["kernel_size"],
                activation=self.layer_config["activation"],
                kernel_initializer=self.layer_config["kernel_initializer"],
                padding=self.layer_config["padding"]
            )
            self.conv_layers.append(layer)

    def build(self, input_shape):
        # Building Intermediate layers:
        for layer in self.conv_layers:
            layer.build((input_shape[0], input_shape[2], input_shape[3], input_shape[4]))

        super().build(input_shape)

    def call(self, inputs):
        stack = []
        for ind in range(self.temporal_dim):
            x = self.conv_layers[ind](inputs[:, ind])
            x = tf.expand_dims(x, axis=1)
            stack.append(x)

        out = Concatenate(axis=1)(stack)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.temporal_dim, input_shape[2], input_shape[3], self.layer_config["filters"])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layer_config": self.layer_config,
                "temporal_dim": self.temporal_dim
            }
        )

        return config







