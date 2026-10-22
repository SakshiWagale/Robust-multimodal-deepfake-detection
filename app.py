"""
app.py
------
Main entry point of the Multimodal Deepfake Detection Flask application.
"""

import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
)

from werkzeug.utils import secure_filename
from config import get_config


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "models/image_classifier.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("==============================================")
print("Model Loaded Successfully!")
print("Model:", MODEL_PATH)
print("Input Shape:", model.input_shape)
print("Output Shape:", model.output_shape)
print("==============================================")


# ============================================================
# CLASSIFICATION THRESHOLD
# ============================================================

# Model output:
# 0 = FAKE
# 1 = REAL
#
# Validation testing showed 0.55 gives the best
# tested overall accuracy among the thresholds checked.

THRESHOLD = 0.45


# ============================================================
# ALLOWED FILE CHECK
# ============================================================

def allowed_file(filename, allowed_extensions):
    """
    Check whether uploaded file extension is allowed.
    """

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in allowed_extensions
    )


# ============================================================
# CREATE FLASK APP
# ============================================================

def create_app(config_name="development"):

    app = Flask(__name__)

    # Load configuration
    app.config.from_object(get_config(config_name))

    # Create upload folder
    os.makedirs(
        app.config["UPLOAD_FOLDER"],
        exist_ok=True
    )


    # ========================================================
    # HOME
    # ========================================================

    @app.route("/")
    def index():

        return render_template(
            "index.html",
            app_name=app.config["APP_NAME"]
        )


    # ========================================================
    # ABOUT
    # ========================================================

    @app.route("/about")
    def about():

        return render_template(
            "about.html",
            app_name=app.config["APP_NAME"]
        )


    # ========================================================
    # DETECT
    # ========================================================

    @app.route("/detect", methods=["GET", "POST"])
    def detect():

        if request.method == "POST":

            # ------------------------------------------------
            # Check file field
            # ------------------------------------------------

            if "media" not in request.files:

                return redirect(
                    url_for("detect")
                )


            # ------------------------------------------------
            # Get uploaded file
            # ------------------------------------------------

            file = request.files["media"]


            # ------------------------------------------------
            # Check filename
            # ------------------------------------------------

            if file.filename == "":

                return redirect(
                    url_for("detect")
                )


            # ------------------------------------------------
            # Check extension
            # ------------------------------------------------

            if file and allowed_file(
                file.filename,
                app.config["ALLOWED_EXTENSIONS"]
            ):

                # --------------------------------------------
                # Secure filename
                # --------------------------------------------

                filename = secure_filename(
                    file.filename
                )


                # --------------------------------------------
                # Create file path
                # --------------------------------------------

                filepath = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    filename
                )


                # --------------------------------------------
                # Save uploaded image
                # --------------------------------------------

                file.save(filepath)

                print()
                print("==============================================")
                print("New Image Uploaded")
                print("File:", filename)
                print("==============================================")


                # =================================================
                # IMAGE PREPROCESSING
                # =================================================

                img = image.load_img(
                    filepath,
                    target_size=(128, 128)
                )


                img_array = image.img_to_array(
                    img
                )


                # Convert pixel values:
                # 0-255 -> 0-1

                img_array = (
                    img_array / 255.0
                )


                # Add batch dimension

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )


                # =================================================
                # MODEL PREDICTION
                # =================================================

                prediction = float(
                    model.predict(
                        img_array,
                        verbose=0
                    )[0][0]
                )


                # =================================================
                # CLASSIFICATION
                # =================================================

                if prediction >= THRESHOLD:

                    verdict = "REAL"

                    confidence = (
                        prediction * 100
                    )

                else:

                    verdict = "FAKE"

                    confidence = (
                        (1.0 - prediction) * 100
                    )


                confidence = round(
                    confidence,
                    2
                )


                # =================================================
                # TERMINAL OUTPUT
                # =================================================

                print("Prediction Score :", prediction)
                print("Threshold         :", THRESHOLD)
                print("Verdict           :", verdict)
                print("Confidence        :", confidence)
                print("==============================================")


                # =================================================
                # RESULT PAGE
                # =================================================

                return render_template(
                    "result.html",

                    app_name=app.config[
                        "APP_NAME"
                    ],

                    filename=filename,

                    verdict=verdict,

                    confidence=confidence,

                    modality="Image Analysis"
                )


        # ========================================================
        # GET REQUEST
        # ========================================================

        return render_template(
            "detect.html",
            app_name=app.config["APP_NAME"]
        )


    # ============================================================
    # RESULT
    # ============================================================

    @app.route("/result")
    def result():

        filename = request.args.get(
            "filename"
        )

        return render_template(
            "result.html",

            app_name=app.config["APP_NAME"],

            filename=filename,

            verdict="No Result",

            confidence=0,

            modality="Image Analysis"
        )


    # ============================================================
    # RETURN APP
    # ============================================================

    return app


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    flask_app = create_app(
        "development"
    )

    flask_app.run(
        host=flask_app.config["HOST"],
        port=flask_app.config["PORT"],
        debug=flask_app.config["DEBUG"],
    )