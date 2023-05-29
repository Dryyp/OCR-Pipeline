from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def spell_correction(text):
    def correct_spelling(input_text):
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(input_ids, max_length=126)
        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_text
    
    model_path = "routes/predict/ocr_pipeline/spelling_correction/BERT/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    corrected_text = correct_spelling(text)
    
    return corrected_text