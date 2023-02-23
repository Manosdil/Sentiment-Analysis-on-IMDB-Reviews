#Necessary Libraries
import pandas as pd 
import numpy as np 
import re
from timeit import default_timer as timer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import openpyxl

#Preprocessing functions
def prepros(review):
    #lower_characters
    def lower_chars(review):
        return review.lower()
    #remove specific patterns
    def remove_patterns(review):
        return re.sub(r'\<br\s\/\>\<br\s\/\>', " ", review)
    #remove urls
    def remove_url(review):
        return re.sub(r"https?\S*\w+", "", review)
    #remove punctuation and special characters
    def remove_punc(review):
        return re.sub(r"[^a-zA-z0-9\s]", "", review)
    #remove extra spaces
    def rem_spaces(review):
        return re.sub(r"\s+", " ", review)
    #strip the review
    def stripit(review):
        return review.strip()
    #lemmatize with WordNetLemmatizer
    def lemmatize(review, wnl=WordNetLemmatizer()):
        return " ".join([wnl.lemmatize(token) for token in word_tokenize(review)])
    #remove stopwords
    def remove_stopwords(review, stopword_list=stopwords.words('english')):
        return " ".join([token for token in word_tokenize(review) if token not in stopword_list])
    return remove_stopwords(lemmatize(stripit(rem_spaces(remove_punc(remove_url(remove_patterns(lower_chars(review))))))))
    
#Vectorizing the review
def tfidf(X_train, X_test, mf = 18000, ngr = 2):
    tf = TfidfVectorizer(max_features = mf, ngram_range=(1,ngr), max_df=24000)
    tf_train = tf.fit_transform(X_train)
    tf_test = tf.transform(X_test)
    return tf_train, tf_test

#Logistic Regression Classifier
def lrc(X_train, X_test, y_train, y_test, c = 1, pen = 'l2'):
    print('\nLogistic Regression')
    lr= LogisticRegression(C = c, penalty= pen )
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\nThe accuracy score in test set is {test_acc*100: .2f}%")
    train_acc = accuracy_score(y_train, lr.predict(X_train))
    print(f"\nThe accuracy score in train set is {train_acc*100: .2f}%\n")
    print(confusion_matrix(y_test, y_pred))

#Linear SV Classifier
def svc(X_train, X_test, y_train, y_test, c = 0.1):
    print('\n\nLinearSVC')
    svc= LinearSVC(C = c)
    svc.fit(X_train, y_train)

    y_pred = svc.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\nThe accuracy score in test set is {test_acc*100: .2f}%")
    train_acc = accuracy_score(y_train, svc.predict(X_train))
    print(f"\nThe accuracy score in train set is {train_acc*100: .2f}%\n")
    print(confusion_matrix(y_test, y_pred))

#Processing the Dataframe
def df_process(df):
    df.drop(['Unnamed: 0', 'type', 'file'], axis = 1, inplace = True)
    df = df[~(df.label == 'unsup')]
    df = pd.get_dummies(df, columns=['label'], drop_first=True)
    return df

#Loading the data into DataFrame
reviews = pd.read_csv('imdb_master.csv', encoding = "ISO-8859-1")
reviews.drop(['Unnamed: 0', 'type', 'file'], axis = 1, inplace = True)
reviews = reviews[~(reviews.label == 'unsup')]
reviews.review = reviews.review.apply(prepros)
reviews = pd.get_dummies(reviews, columns=['label'], drop_first=True)
#reviews.to_excel('ForModel.xlsx')
#Splitting the reviews to train and test
X_train, X_test, y_train, y_test = train_test_split(reviews.review, 
                reviews.label_pos.values, test_size = 0.2, random_state = 42)

tf_train, tf_test = tfidf(X_train, X_test)
lrc(tf_train, tf_test, y_train, y_test)
svc(tf_train, tf_test, y_train, y_test)