import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


img_height, img_width = 224, 224


model = load_model("best_model.h5")
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def predict(img_path):
    try:
        img = image.load_img(img_path, target_size=(img_height, img_width))
    except:
        print(f"❌ Image not found: {img_path}")
        return

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class = labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    print(f"🖼️ Image: {img_path}")
    print(f"✅ Predicted: {predicted_class} ({confidence:.2f}%)")

predict("dataset/test/cat/cat1.jpg")
predict("dataset/test/dog/dog1.jpg")
