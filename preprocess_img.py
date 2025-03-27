import tensorflow as tf

def Preprocess_image(img_path="img.jpg"):
    img=tf.io.read_file(img_path)
    img=tf.io.decode_jpeg(img,channels=3)
    img=tf.image.resize(img,[256,256])
    img/=255.0
    img=tf.expand_dims(img,axis=0)
    return img


if __name__=="__main__":
    print(Preprocess_image(img_path="img.jpg").shape)