# from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
def preprocess_image(path):
    
    image = Image.open(path).convert('RGB') 
    # Resize the image
    img = image.resize((32, 32))
    # Convert the image to an array
    img_array = np.array(img)
    # Normalize pixel values to the range [0, 1]
    img_array = img_array / 255.0
    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, image