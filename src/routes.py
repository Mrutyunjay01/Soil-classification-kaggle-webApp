import os
from flask import request, render_template
from keras.models import load_model

from src.utils import model_predict
from app import app

model_path = "weights/model_v1.h5"
model = load_model(model_path)

@app.route("/")
@app.route("/predict", methods=["GET"])
def index():
    return render_template("index.html")
    pass


@app.route("/predict", methods=["GET", "POST"])
def predict():
    print("Entered")
    # fetch input
    file = request.files["image"]
    filename = file.filename
    print("@@ Input Posted = ", filename)

    file_path = os.path.join("../static/user_uploaded", filename)
    file.save(file_path)

    print("@@ Predicting class...")
    pred, output_page = model_predict(file_path, model)

    return render_template(output_page, pred_output=pred, user_image=file_path)
    pass
