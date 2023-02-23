import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

def svc_model():
    def tfidf(X_train, X_test, mf = 18000, ngr = 2):
        tf = TfidfVectorizer(max_features = mf, ngram_range=(1,ngr), max_df = 24000)
        tf_train = tf.fit_transform(X_train)
        return tf , tf_train

    #Linear SV Classifier
    def svc(X_train, y_train, c = 0.1):
        svc= LinearSVC(C = c)
        svc.fit(X_train, y_train)
        return svc

    reviews = pd.read_excel('ForModel.xlsx')
    X_train, X_test, y_train, y_test = train_test_split(reviews.review, 
                    reviews.label_pos.values, test_size = 0.2, random_state = 42)
    tf, tf_train = tfidf(X_train, X_test, mf = 18000, ngr = 2)
    model = svc(tf_train, y_train, c = 0.1)
    return tf, model
