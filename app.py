from flask import (
    Flask,
    render_template,
    request,
    url_for
)
import os
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
from utils.image_processing import preprocess_image


app = Flask(__name__)
model = load_model('model/cifar10.h5')

class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('home.html', prediction='No file uploaded')

    file = request.files['image']
    
    if file.filename == '':
        return render_template('home.html', prediction='No file uploaded')
    
    img_array, img = preprocess_image(file)
    
    prediction = model.predict(img_array).argmax()
    
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    img.save(img_path)
    
    return render_template('home.html', prediction=class_name[prediction], img_url=img_path)

if __name__ == '__main__':
    app.run(debug=True)