import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import base64
import cv2

import dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html

app = dash.Dash(__name__)

model = tf.keras.models.load_model('./image_classification_model')

app.layout = html.Div([
    html.Img(id='image-display', 
        style={
            'width': '100px',
            'height': '100px',
            'lineHeight': '60px',
            'textAlign': 'center',
            'margin': '10px'
        },),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '500px',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload', 
        style={
            'width': '500px',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },),
], style={'backgroundColor': "#ffffff"})

@app.callback(Output('image-display', 'src'),
              Input('upload-image', 'contents'))
def set_image(image):
    if image is None:
        raise dash.exceptions.PreventUpdate()
    return image[0]

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'))
def get_prediction(image):
    if image is None:
        raise dash.exceptions.PreventUpdate()
    image = image[0].split(',')[1]
    decoded = base64.b64decode(image)
    np_data = np.fromstring(decoded, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.invert(img_gray)
    img_gray = img_gray / 255.0
    plt.imshow(img_gray)
    confidence = model.predict(np.array([img_gray])).tolist()[0]
    max_confidence = max(confidence)
    return f"The model has predicted the image to be: {confidence.index(max_confidence)}"


if __name__ == "__main__":
    app.run_server(debug=True)