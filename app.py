
from tf_keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2

# Load model and labels
print("Loading model...", flush=True)
model = load_model("keras_Model_fixed.h5", compile=False)
print("Model loaded!", flush=True)
class_names = open("labels.txt", "r").readlines()

# Open webcam
camera = cv2.VideoCapture(0)
print("Webcam opened:", camera.isOpened(), flush=True)

while True:
  success, frame = camera.read()

  if not success:
    break

  # Convert webcam frame to image
  image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  image = Image.fromarray(image)

  # Resize image to 224x224
  image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

  # Convert image to array and normalize
  image_array = np.asarray(image)
  normalized_image = (image_array.astype(np.float32) / 127.5) - 1

  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
  data[0] = normalized_image

  # Predict class
  prediction = model.predict(data, verbose=0)

    # Show all classes and their confidence
  for i in range(len(class_names)):
    class_name = class_names[i][2:].strip()
    confidence = prediction[0][i] * 100

    text = f"{class_name}: {confidence:.1f}%"

    cv2.putText(
        frame, text, (20, 40 + i * 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1,
        (0, 255, 0), 2
    )

  cv2.imshow("Webcam Classification", frame)

  # Press Q to stop
  if cv2.waitKey(1) & 0xFF == ord("x"):
    break
  
  if cv2.getWindowProperty(
    "Webcam Classification",
    cv2.WND_PROP_VISIBLE
  ) < 1:
    break

camera.release()
cv2.destroyAllWindows()
