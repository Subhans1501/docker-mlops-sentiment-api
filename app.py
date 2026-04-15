from flask import Flask, request, jsonify, render_template_string
import pickle
import os
app=Flask(__name__)
MODEL_PATH='model/model.pkl'
VECTORIZER_PATH='model/vectorizer.pkl'
model=pickle.load(open(MODEL_PATH, 'rb'))
vectorizer=pickle.load(open(VECTORIZER_PATH, 'rb'))
HTML_PAGE="""
<!DOCTYPE html>
<html>
<head>
    <title>AI Sentiment Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f9; padding: 50px; text-align: center; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0px 0px 15px rgba(0,0,0,0.1); max-width: 500px; margin: auto; }
        textarea { width: 100%; padding: 10px; margin-bottom: 15px; border-radius: 5px; border: 1px solid #ccc; }
        button { background-color: #28a745; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background-color: #218838; }
        .result { margin-top: 20px; font-weight: bold; font-size: 18px; }
        .positive { color: green; }
        .negative { color: red; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Customer Review Analyzer</h2>
        <form action="/predict_web" method="POST">
            <textarea name="text" rows="4" placeholder="Type a review here..." required></textarea><br>
            <button type="submit">Analyze Sentiment</button>
        </form>
        {% if prediction %}
            <div class="result">
                Result: <span class="{{ prediction }}">{{ prediction | upper }}</span>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""
@app.route('/',methods=['GET'])
def home():
    return render_template_string(HTML_PAGE)
@app.route('/predict_web',methods=['POST'])
def predict_web():
    text=request.form['text']
    text_vec=vectorizer.transform([text])
    prediction=str(model.predict(text_vec)[0]) 
    return render_template_string(HTML_PAGE, prediction=prediction)
@app.route('/predict', methods=['POST'])
def predict_api():
    data=request.json
    text=data.get('text', '')
    text_vec=vectorizer.transform([text])
    prediction=str(model.predict(text_vec)[0]) 
    return jsonify({"input_text": text, "predicted_sentiment": prediction})
if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)