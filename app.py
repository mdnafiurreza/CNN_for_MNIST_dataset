import numpy as np
from flask import Flask, request, render_template, jsonify
import onnxruntime as ort
from PIL import Image

app = Flask(__name__)


ort_session = ort.InferenceSession("mnist_cnn.onnx")

def preprocess_image(image):

    image = image.convert("L")
    image = image.resize((28, 28))
    image_array = np.array(image).astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0) 
    image_array = np.expand_dims(image_array, axis=0) 
    return image_array

def predict_digit(image):
    
    processed = preprocess_image(image)
    ort_inputs = {ort_session.get_inputs()[0].name: processed}
    preds = ort_session.run(None, ort_inputs)[0][0]
    pred_digit = int(np.argmax(preds))
    confidence = preds.tolist()
    return pred_digit, confidence

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        img = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": "File is not a valid image"}), 400

    pred_digit, confidence = predict_digit(img)
    response = {
        "predicted_digit": pred_digit,
        "confidence_scores": confidence
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)


