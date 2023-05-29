from .text_detection.EAST_text_detection import EAST
from .text_recognition.inference_model import ImageToWordModel
from .spelling_correction.spell_correction import spell_correction

def ocr(image):
    vocab = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    model_path = 'routes/predict/ocr_pipeline/text_recognition/model/'
    model = ImageToWordModel(model_path=model_path, char_list=vocab)

    rois, image, orig = EAST(image)
    
    words = []
    for roi in rois:
        predicted_text = model.predict(roi)
        words.append(predicted_text)
    
    corrected_text = spell_correction(" ".join(words))
    
    return [image, orig, " ".join(words), corrected_text]
