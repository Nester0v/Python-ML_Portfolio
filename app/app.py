from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = load_model('models/brain_tumor_model.keras')

# Define class labels
CLASS_LABELS = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
CLASS_LABELS_READABLE = {
    'glioma_tumor': 'Glioma Tumor',
    'meningioma_tumor': 'Meningioma Tumor',
    'no_tumor': 'Tumors not found',
    'pituitary_tumor': 'Pituitary Tumor'
}

@app.route('/')
def index():
    # Render the upload and prediction page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the file to the upload directory
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        uploaded_image_url = url_for('static', filename=f'uploads/{file.filename}')

        # Preprocess the uploaded image for model prediction
        img = image.load_img(file_path, color_mode="grayscale", target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Expand dims to match model's input shape

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = CLASS_LABELS[np.argmax(predictions)]
        predicted_class_readable = CLASS_LABELS_READABLE[predicted_class]
        confidence = np.max(predictions)

        confidence_percentage = round(confidence * 100, 2)

        # Return the prediction, confidence, and image URL to the webpage
        return render_template('index.html',
                               prediction=predicted_class_readable,
                               confidence=confidence_percentage,
                               image=uploaded_image_url)

if __name__ == "__main__":
    # Run the app
    app.run(debug=True)
