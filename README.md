# Student Focus and Distraction Classification

This project uses **Google Teachable Machine** to train an image classification model that predicts whether a student appears focused or distracted based on webcam images.

The trained model is exported in Keras format and used in a Python application with OpenCV for real-time webcam classification.

The predictions represent visible behavior and do not necessarily reflect a student's actual attention.

## Technologies Used

- Google Teachable Machine — model training
- Python — application development
- TensorFlow / Keras — loading the trained model
- OpenCV — webcam access and displaying predictions
- NumPy — image processing and prediction results
- Pillow — image resizing and preprocessing

## Project Files

```text
ml_tugas/
├── app.py
├── fix_model.py
├── keras_Model.h5
├── keras_Model_fixed.h5
├── labels.txt
├── requirements.txt
└── README.md
```

| File | Description |
|---|---|
| `app.py` | Main application for real-time webcam classification |
| `fix_model.py` | Fixes compatibility issues in the exported Keras model |
| `keras_Model.h5` | Original trained model exported from Teachable Machine |
| `keras_Model_fixed.h5` | Modified model compatible with the application's Keras environment |
| `labels.txt` | Contains the names of the classification classes |
| `requirements.txt` | Lists the Python libraries required to run the application |

## Why Is `fix_model.py` Needed?

The original model was exported from Google Teachable Machine using an older Keras format.

When loading it with newer Keras versions, an error occurred:

```text
Unrecognized keyword arguments passed to DepthwiseConv2D: {'groups': 1}
```

The `fix_model.py` script creates a copy of the original model and removes the unsupported `groups` argument from the affected layer configurations.

This produces `keras_Model_fixed.h5` without retraining the model or intentionally changing its learned weights.

We also use the legacy `tf-keras` library to handle compatibility with the exported model.

## How to Run

**1. Create a virtual environment (Python 3.12)**

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate
```

**2. Install the required libraries**

```cmd
pip install -r requirements.txt
```

**3. Fix the model (if the fixed model is not already available)**

```cmd
python fix_model.py
```

**4. Start the application**

```cmd
python app.py
```

The webcam window will open and display the classification results, including the prediction percentages for both classes.

## How to Use

1. Allow the application to access the webcam.
2. Sit in front of the camera.
3. Perform different study-related activities.
4. Observe the predicted classes and percentages.
5. Press **x** or close the webcam window to exit.

## How It Works

1. OpenCV captures an image from the webcam.
2. The image is converted from BGR to RGB.
3. Pillow resizes and crops it to 224 × 224 pixels.
4. NumPy converts the image into an array and normalizes its pixel values.
5. The trained Keras model predicts the classification scores.
6. OpenCV displays both class names and their prediction percentages on the webcam.
