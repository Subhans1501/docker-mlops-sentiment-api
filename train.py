import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
print("Loading dataset from data/train.csv...")
df=pd.read_csv('data/train.csv', nrows=5000)
text_column='content'      
label_column='label'  
X=df[text_column].values
y=df[label_column].values
print("Training the Naive Bayes model...")
vectorizer=CountVectorizer(max_features=3000)
X_vec=vectorizer.fit_transform(X)
model=MultinomialNB()
model.fit(X_vec,y)
os.makedirs('model',exist_ok=True)
pickle.dump(model,open('model/model.pkl','wb'))
pickle.dump(vectorizer,open('model/vectorizer.pkl','wb'))
print("Success: Model trained on real data and saved to 'model/'")