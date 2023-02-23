import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def review_process(review, stopwords_list = stopwords.words('english'), wnl=WordNetLemmatizer()):
    review = review.lower()
    review = re.sub(r"[^a-zA-z0-9\s]", "", review)
    review = re.sub(r"\s+", " ", review)
    review = ([wnl.lemmatize(token) for token in word_tokenize(review)])
    review =  " ".join([token for token in review if token not in stopwords_list])
    return review.strip()
    
    