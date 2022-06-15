import nltk
import joblib
from nltk.corpus import stopwords
import re
import os
import pickle
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# load pickled models
filename1 = 'static/model2.pkl'
filename2 = 'static/tfidfvect2.pkl'
model = pickle.load(open(filename1, 'rb'))
tfidfvect = pickle.load(open(filename2, 'rb'))

def predictor(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    prediction = 'FAKE' if model.predict(review_vect) == 0 else 'REAL'
    return prediction
