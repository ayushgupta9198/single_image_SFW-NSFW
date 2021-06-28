import os
from nsfw_detector import predict 
from nsfw_detector.predict import thresold_calculation

model = predict.load_model('./mobilenet_v2_140_224')

def image_safty_check(img_path=None):
    """
    image_safty_check functio varify that image has no adult content by utilizing nsfw_detector model
    
    >>>img_path ="Path to the image"   

    """
    output = predict.classify(model, img_path)
    result = thresold_calculation(output)
    return result


if __name__ == '__main__':
    path="/home/ayush-ai/Music/Project/7.jpg"   
    res = image_safty_check(path)
    print("Image is safe :", res)