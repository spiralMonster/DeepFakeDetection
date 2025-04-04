import tensorflow as tf
from tensorflow.keras.layers import Layer,MaxPooling2D,Concatenate

class MaxPoolingVideo(Layer):
    def __init__(self, pool_size, padding, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.padding = padding

    def call(self, inputs):
        stack = []
        for ind in range(inputs.shape[1]):
            x = MaxPooling2D(
                pool_size=self.pool_size,
                padding=self.padding
            )(inputs[:, ind])
            x = tf.expand_dims(x, axis=1)
            stack.append(x)

        out = Concatenate(axis=1)(stack)
        return out

    def compute_output_shape(self, input_shape):
        out_img_width = input_shape[2] // self.pool_size[0]
        out_img_height = input_shape[3] // self.pool_size[1]
        return (input_shape[0], input_shape[1], out_img_width, out_img_height, input_shape[4])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pool_size": self.pool_size,
                "padding": self.padding
            }
        )
        return config
