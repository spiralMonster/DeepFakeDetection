import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from Models.conv2plus1d import Conv2Plus1D
from Models.max_pooling_video import MaxPoolingVideo
from Models.optical_flow_classifier import OpticalFlowClassifier
from VideoPreprocessing.get_frames_from_video import GetFrameFromVideo
from VideoPreprocessing.get_optical_flow_frames import GetOpticalFlowFrames

def DeepFakeVideoDetection(video_path):
    result={}
    #Loading Model:
    with CustomObjectScope(
            {
                "Conv2Plus1D":Conv2Plus1D,
                "MaxPoolingVideo":MaxPoolingVideo,
                "OpticalFlowClassifier":OpticalFlowClassifier
            }
    ):
        model=load_model("deepfake_video_detection.h5")


    model.summary()
    print("Model Loaded...")

    #Data For prediction:
    conv_inp=GetFrameFromVideo(
        video_path=video_path
    )
    optical_flow_inp=GetOpticalFlowFrames(frame_data=conv_inp[0])

    print(f"Conv Input shape:{conv_inp.shape}")
    print(f"Optical Flow Input shape:{optical_flow_inp.shape}")

    print("Data preprocessed..")

    #Making Prediction:
    prediction=model.predict([conv_inp,optical_flow_inp],batch_size=1)
    pred=float(prediction[0])
    print("Predictions made")

    if pred<0.5:
        result["verdict"]="Deepfake Video"
        result["confidence"]=1-pred

    else:
        result["verdict"]="Real Video"
        result["confidence"]=pred


    print("Result sent..")
    print(f"Result:{result}")
    return result




