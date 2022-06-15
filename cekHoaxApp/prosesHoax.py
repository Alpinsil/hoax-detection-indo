import pickle
from lib2to3.pgen2 import token
import pandas as pd
import numpy as np
import nltk
from cekHoaxApp import *
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from cekHoaxApp.preprocessing import Preprocessing

def antiHoax(text, jenis):
    # # Membuat Stemmer
    # factory = StemmerFactory()
    # stemmer = factory.create_stemmer()

    # # Label Encoder use to Encode target labels with value between 0 and n_classes-1
    # Encoder = LabelEncoder()

    # # TfidfVectorizer Convert a collection of raw documents to a matrix of TF-IDF features.
    # Tfidf_vect = TfidfVectorizer()

    # # data_train = pd.read_csv('./Data_latih.csv')
    # data_train = pd.read_csv('static/Data_latih.csv')
    # data_train['label'].value_counts()

    # # Dari cell sebelumnya terlihat jelas bahwa dataset kita sangat tidak balance
    # # Untuk membuat datasetnya balance
    # # Pilih dataset dengan label 1 dan lakukan randomisasi untuk setiap baris datanya
    # false_news = data_train[data_train['label'] == 1].sample(frac=1)

    # # Concat dataset berlabel 1 yang telah dipilih dengan dataset berlabel 0
    # # dimana jumlah dataset berlabel 1 yang digabungkan sejumlah banyak dataset berlabel 0 + 200
    # true_fact = data_train[data_train['label'] == 0]
    # df = true_fact.append(false_news[:len(true_fact) + 200])

    # # menggunakan fitur narasi dalam melakukan prediksi terhadap label
    # feature = df['narasi']
    # label = df['label']

    # # Mengubah semua huruf pada setiap baris menjadi huruf kecil dan
    # # melakukan stemming pada setiap baris
    # lower = [stemmer.stem(row.lower()) for row in feature]

    # # Hasil stem dan lower
    # lower[:5]

    # # Melakukan tokenisasi untuk setiap baris dataset
    # tokens = [word_tokenize(element) for element in lower]

    # # Hasil tokenisasi setiap baris
    # tokens[:5]

    # # train_test_split digunakan untuk memecah dataset menjadi 2 bagian
    # # X_train dan y_train mewakili data yang akan dilakukan pada fitting model(Training model)
    # # X_test dan y_test  mewakili data yang akan dilakukan pada evaluasi model
    # X_train, X_test, y_train, y_test = train_test_split(tokens, label, test_size=0.2, stratify=label)

    # # Melihat ukuran data latih dan data uji
    # print('X_train : ', len(X_train))
    # print('X_test : ', len(X_test))

    # y_train = Encoder.fit_transform(y_train)
    # y_test = Encoder.fit_transform(y_test)

    # print(y_train)

    # Tfidf_vect.fit(["".join(row) for row in X_train])

    # X_train_Tfidf = Tfidf_vect.transform([" ".join(row) for row in X_train])
    # X_test_Tfidf = Tfidf_vect.transform([" ".join(row) for row in X_test])

    # # Classifier - Algorithm - SVM
    # # fitting/training datasets pada algoritma SVM(Support Vector Machine)
    # SVM = svm.SVC(C=1.0, kernel='linear', degree=1, gamma="auto", verbose=True)
    # SVM.fit(X_train_Tfidf, y_train)  # predict the labels on validation dataset

    # # Menggunakan metrics accuracy untuk melihat performa model
    # predictions_SVM = SVM.predict(X_test_Tfidf)
    # print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, y_test)*100)

    # rf = RandomForestClassifier()
    # rf.fit(X_train_Tfidf, y_train)

    # prediction_rf = rf.predict(X_test_Tfidf)
    # print("RandomForest Accuracy Score -> ", accuracy_score(prediction_rf, y_test)*100)

    #save model
    filename = 'static/finalized_model.pkl'
    # pickle.dump(prediction_rf, open(filename, 'rb'))
    d = pickle.load(open(filename, 'rb'))
    # d.predict([text])
    # print(d)
    # print(text)
    # return 0
    return d


def proses(text, jenis):
    hasil = antiHoax(Preprocessing(text).run(), jenis); 
    return hasil