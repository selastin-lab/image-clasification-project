from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)


img_height, img_width = 224, 224

model = load_model("best_model.h5")
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class = labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            predicted_class, confidence = predict_image(filepath)
            return f"Predicted: {predicted_class} ({confidence:.2f}%)"
    return """
    <h2>Upload an image</h2>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit">
    </form>
    """

if __name__ == "__main__":
    app.run(debug=True)
