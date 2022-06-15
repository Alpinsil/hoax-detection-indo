import nltk
import string

from nltk.stem.porter import *
from langdetect import detect
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

class Preprocessing():
    
    kalimat = ''

    def __init__(self, data):
        self.data = data
        Preprocessing.kalimat = self.data

    def casefold(self):
        return self.data.casefold()

    def punctuation(self):
        return self.data.translate(str.maketrans('','',string.punctuation))

    def removeNumber(self):
        return re.sub(r"\d+", "", self.data)

    def tokenizing(self):
        return nltk.word_tokenize(self.data)

    def stemming(self):
        # print('\n=============proses Steming=============')
        # print(detect(Preprocessing.kalimat))
        # if detect(Preprocessing.kalimat) == 'id':
            # print(len(self.data))
        ps = StemmerFactory()
        stemmer = ps.create_stemmer()
        res = [None] * len(self.data)
        i = 0
        for w in self.data:
            res[i] = stemmer.stem(w)
            i+=1
            # print(res[i])
        return res
        # else:
        #     ps = PorterStemmer()
        #     res = [None] * len(self.data)
        #     i = 0
        #     for w in self.data:
        #         res[i] = ps.stem(w)
        #         i+=1
        #     return res

    def filtering(self):
        if detect(Preprocessing.kalimat) == 'id':
            listStopword =  set(stopwords.words('indonesian'))
        else:
            listStopword =  set(stopwords.words('english'))

        res = []
        for t in self.data:
            if t not in listStopword:
                res.append(t)
        return res
    
    def run(self):
        self.data = self.punctuation()
        # print("\n=======================punctuation====================")
        # print(self.data)
        self.data = self.casefold()
        # print("\n=======================casefolding====================")
        # print(self.data)
        self.data = self.removeNumber()
        # print("\n=======================remove number====================")
        # print(self.data)
        self.data = self.tokenizing()
        # print("\n=======================tokenizing====================")
        # print(self.data)
        self.data = self.stemming()
        # print("\n=======================stemming====================")
        # print(self.data)
        self.data = self.filtering()
        # print("\n=======================filtering====================")
        # print(self.data)
        return self.data

