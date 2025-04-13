# MNIST Digit Classifier Web App with PyTorch and Flask

## Overview
This is a project to build, train, and deploy a Convolutional Neural Network for classifying handwritten digits from the MNIST dataset using PyTorch. The trained model is exported to the ONNX format and then deployed through a Flask web application. The web application allows users to upload an image from the sample_images folder (or select a sample) and receive a digit prediction along with a confidence chart as results.


## Installation


Install the dependencies using the requirements.txt file:
   pip install -r requirements.txt


## Run the Flask Web Application

1. To start the Flask Server:

   In terminal, run:
       python app.py

2. Access the Web App:

   Open browser and navigate to http://127.0.0.1:5000/

3. To use the Web App:

   - Image Upload: Click the "Choose File" button to select an image file of a handwritten digit and then click on the button "Predict" to obtain the predicted digit and confidence scores.
   - Sample Images : Shows the sample images provided in the sample_images folder.



## Md Nafiur Reza

