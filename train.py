from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
X = ["Great product,highly recommended","Terrible service,very disappointed","I love this application","Worst experience ever, do not buy"]
y=["positive","negative","positive","negative"]
vectorizer=CountVectorizer()
X_vec=vectorizer.fit_transform(X)
model=MultinomialNB()
model.fit(X_vec,y)
os.makedirs('model',exist_ok=True)
pickle.dump(model, open('model/model.pkl','wb'))
pickle.dump(vectorizer,open('model/vectorizer.pkl','wb'))
print("Success: NLP Sentiment Model and Vectorizer trained and saved in 'model/' directory!")