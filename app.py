import argparse
import io
import os
from PIL import Image, ImageDraw, ImageFont
import torch
from flask import Flask, render_template, request, redirect
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from tensorflow.keras.models import load_model
import urllib.request
from werkzeug.utils import secure_filename
import datetime

app = Flask(__name__)
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

@app.route("/detection", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return redirect(request.url)

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model([img])

        results.render()  # updates results.imgs with boxes and labels

        # Count 
        count_rusak_berat  = results.pred[0][:, -1].eq(0).sum().item()
        count_rusak_ringan = results.pred[0][:, -1].eq(1).sum().item()
        count_rusak_sedang  = results.pred[0][:, -1].eq(2).sum().item()
        count_total =  count_rusak_berat + count_rusak_ringan + count_rusak_sedang

        # Add count labels to the image
        img_with_labels = Image.fromarray(results.ims[0])
        draw = ImageDraw.Draw(img_with_labels)
        font = ImageFont.load_default()

        # Label for total items in red
        draw.text((10, 20), f"Total Kerusakan: {count_total}", fill="red", font=font)
    
        results.render()  # updates results.imgs with boxes and labels
        now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        img_savename = f"static/{now_time}.png"
        Image.fromarray(results.ims[0]).save(img_savename)
        img_with_labels.save(img_savename)
        return redirect(img_savename)
    
    return render_template('detection.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/regform')
def regform():
    return render_template('regform.html')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLOv5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port)