from flask import Blueprint, request, jsonify
import numpy as np
import cv2
import base64

predict = Blueprint('predict', __name__)

@predict.route('/predict', methods=['POST'])
def prediction():
    if 'image' not in request.files:
        return {'error': 'No file selected'}
    
    file = request.files['image']
    img_bytes = file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    from .ocr_pipeline.ocr import ocr
    document = ocr(image)
    
    _, buffer_preprocessed = cv2.imencode('.jpg', document[0])
    preprocessed_base64 = base64.b64encode(buffer_preprocessed).decode('utf-8')
    
    _, buffer_orig = cv2.imencode('.jpg', document[1])
    orig_base64 = base64.b64encode(buffer_orig).decode('utf-8')
    
    images = [preprocessed_base64, orig_base64]
    
    texts = [document[2], document[3]]
    
    return jsonify({'images': images, 'texts': texts})