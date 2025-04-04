import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from Model.deepfake_model import DeepFakeDetectionModel
from Model.inception_layer import Inception_cell
from Model.skip_connection_layer import SkipConnection
from get_image import GetImage
from preprocess_img import Preprocess_image

def DeepFakeDetection(img_url):
    img_path=GetImage(url=img_url)

    if img_path:
        img=Preprocess_image(img_path)
        print(img.shape)
        #Load Model:
        with CustomObjectScope({
            "Inception_cell":Inception_cell,
            "SkipConnection":SkipConnection
        }):
            model=load_model("deepfake_model.h5")

        model.summary()
        predictions=model.predict(img,batch_size=1)

        response={}
        response["deepfake_confidence"]=predictions[0][0]
        response["real_image_confidence"]=predictions[0][1]

        final_pred=tf.argmax(predictions,axis=1)[0]
        if final_pred==0:
            response["final_verdict"]="Deepfake Image"

        else:
            response["final_verdict"]="Real Image"

        return response

    else:
        print("Image not retrieved...")
        return {}


if __name__=="__main__":
    print(DeepFakeDetection(img_url="https://factfury-file.s3.amazonaws.com/uploads/1742668996004-1742668995981-adhaar.jpg"))


