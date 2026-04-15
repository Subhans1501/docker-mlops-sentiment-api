from flask import Flask, request, jsonify
import pickle
import os
app=Flask(__name__)
MODEL_PATH='model/model.pkl'
VECTORIZER_PATH='model/vectorizer.pkl'
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = pickle.load(open(MODEL_PATH, 'rb'))
    vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
else:
    raise FileNotFoundError("Model files not found. Please run train_model.py first.")
@app.route('/',methods=['GET'])
def health_check():
    return jsonify({
        "status": "active",
        "service": "NLP Sentiment Inference API",
        "version": "1.0"
    })
@app.route('/predict',methods=['POST'])
def predict():
    try:
        data=request.json
        text=data.get('text','')
        if not text:
            return jsonify({"error":"No text provided"}),400
        text_vec=vectorizer.transform([text])
        prediction=model.predict(text_vec)[0]
        return jsonify({
            "input_text":text, 
            "predicted_sentiment":prediction
        })
    except Exception as e:
        return jsonify({"error":str(e)}),500
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)