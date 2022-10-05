import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2

import gradio as gr

model = tf.keras.models.load_model('./image_classification_model')

def get_prediction(image):
  img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  img_gray = np.invert(img_gray)
  img_gray = img_gray / 255.0
  plt.imshow(img_gray)
  confidence = model.predict(np.array([img_gray])).tolist()[0]
  max_confidence = max(confidence)
  return confidence.index(max_confidence)

demo = gr.Interface(fn=get_prediction, inputs="image", outputs="text")

demo.launch()