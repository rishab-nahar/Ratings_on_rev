# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

hotel_reviews = pd.read_csv("hotel_review.csv")
corpus = []

req = hotel_reviews.iloc[:, [1, 14, 15]].values
rating = hotel_reviews.iloc[:, 13].values
#taking care of missing data
from sklearn.impute import SimpleImputer

missingvalues = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=1)
rating = missingvalues.fit_transform(rating.reshape(-1, 1))
#creating the bag of words model
for i in range(35912):
    review = re.sub("[^a-zA-Z]", " ", str(req[i][1]) + " " + str(req[i][2])).lower().split()
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    stop_words.remove("not")
    stop_words.add("nan")
    review = [ps.stem(word) for word in review if not word in stop_words]
    review = ' '.join(review)
    corpus.append(review)

#creating the sparse matrix
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=4000)
X = cv.fit_transform(corpus).toarray()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, rating, test_size=0.15)


from sklearn.tree import DecisionTreeRegressor
#fitting the regressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
pred_rating = regressor.predict(X_test)

accuracy = len(y_test)
for i in range(len(y_test)):
    if abs(y_test[i] - pred_rating[i]) > 0.5:
        accuracy -= 1
print("accuracy for the model ={}".format((accuracy / len(y_test)) * 100))

