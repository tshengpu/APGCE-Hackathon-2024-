import os
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
import pickle
import uuid

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the pre-trained model from the pickle file
model_path = 'model.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Function to check if the file is an allowed image type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to preprocess the image before prediction
def preprocess_image(image_path):
    # Open the image file
    img = Image.open(image_path).convert('RGB')
    # Resize to the input size your model expects (example: 224x224)
    img = img.resize((224, 224))
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Normalize the image (example: scaling pixel values to 0-1)
    img_array = img_array / 255.0
    # Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']

        if file.filename == '':
            return "No selected file", 400

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = str(uuid.uuid4()) + "_" + file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image and make predictions
            img_array = preprocess_image(file_path)
            prediction = model.predict(img_array)
            
            # You may need to adjust this depending on your model's output
            predicted_class = np.argmax(prediction, axis=1)[0]

            return render_template('index2.html', filename=filename, result=predicted_class)

    return render_template('index2.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    # Make sure the upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
