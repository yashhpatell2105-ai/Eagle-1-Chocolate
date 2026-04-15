from flask import Flask, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image

app = Flask(__name__)
CORS(app, origins="*")  # Allow all origins

# Load your trained YOLO model
model = YOLO("Eagle-1(Chocolate)\my_model.pt")

@app.route('/detect', methods=['POST', 'OPTIONS'])
def detect():
    if request.method == 'OPTIONS':
        return '', 200  # Handle preflight request

    file = request.files['image']
    img_bytes = file.read()

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(frame)
    annotated_frame = results[0].plot()

    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(annotated_rgb)

    img_io = io.BytesIO()
    pil_img.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')