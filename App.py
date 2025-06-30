from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("model/poultry_model.h5")

# Make sure this matches your dataset class folder names
class_labels = ['Coccidiosis', 'New Castle', 'Healthy', 'salmonella']

# Ensure upload directory exists
UPLOAD_FOLDER = 'static/uploaded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]

        return render_template('result.html', prediction=predicted_class, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)

from sklearn.metrics import classification_report, confusion_matrix

val_generator.reset()
preds = model.predict(val_generator)
y_pred = np.argmax(preds, axis=1)
y_true = val_generator.classes

print("Classification Report:\n", classification_report(y_true, y_pred, target_names=list(val_generator.class_indices.keys())))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

from tensorflow.keras.models import load_model
import numpy as np
import cv2

model = load_model("model/poultry_model.h5")
img_path = "static/uploaded/image1.jpg"  # Replace with actual test image path

img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224)) / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
predicted_class = np.argmax(prediction)

class_labels = ['Coccidiosis', 'New Castle', 'Healthy', 'salmonella']
print("Predicted Class:", class_labels[predicted_class])
