import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import base64
import cv2

from flask import Flask, render_template, request, jsonify


app = Flask(__name__, static_url_path="", static_folder="resources/static",
            template_folder="resources/templates")

model = tf.keras.models.load_model('./image_classification_model')

def get_prediction(image):
  decoded = base64.b64decode(image)
  np_data = np.fromstring(decoded, np.uint8)
  img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_gray = np.invert(img_gray)
  img_gray = img_gray / 255.0
  plt.imshow(img_gray)
  confidence = model.predict(np.array([img_gray])).tolist()[0]
  max_confidence = max(confidence)
  return confidence.index(max_confidence)

@app.route('/')
def home():
    return 'Home Page.'


@app.route('/ml', methods=['GET', 'POST'])
def mlpage():
    if request.method == 'POST':
        image = request.form.get('image')
        prediction = get_prediction(image)
        return jsonify({prediction: prediction})
    return render_template('ml.html')


if __name__ == '__main__':
    app.run(debug=True)
