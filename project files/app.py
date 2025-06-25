import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Define class names
CLASS_NAMES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# Model path
MODEL_PATH = 'Blood Cell.h5'
model = load_model(MODEL_PATH)

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make prediction
        img = image.load_img(filepath, target_size=(224, 224)) # MobileNetV2 default input size
        x = image.img_to_array(img)
        x = x / 255.0 # Rescale pixel values to 0-1
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        
        pred = model.predict(images)
        predicted_class = CLASS_NAMES[np.argmax(pred)]
        result = predicted_class
        
        # Get relative path for the template
        relative_path = os.path.join('uploads', filename)
        
        return render_template('result.html', result=result, image_path=relative_path)

if __name__ == '__main__':
    app.run(debug=True) 