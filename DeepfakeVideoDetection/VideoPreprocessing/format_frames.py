import tensorflow as tf

def FormatFrames(frame,output_shape):
    img=tf.image.convert_image_dtype(frame,tf.float32)
    img=tf.image.resize(img,output_shape)
    return img