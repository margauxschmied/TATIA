import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords, webtext
from nltk.stem import WordNetLemmatizer
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.probability import FreqDist
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from ast import literal_eval
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB

stopwords_list = stopwords.words('english')
stopwords_list += list(string.punctuation)
stopwords_list += ['one', 'two', 'three', 'four', 'five', 'six', 'seven',
                   'eight', 'nine', 'go', 'goes', 'get', 'also', 'however', 'tells']
stopwords_list += [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#stopwords_list += ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',]
stopwords_list += [x for x in string.ascii_lowercase]


def clean_text(text):
    text = text.lower()

    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub('\d+', ' ', text)

    text = text.strip(' ')
    text = nltk.word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    stem_sentence = []
    for word in text:
        stem = lemmatizer.lemmatize(word)
        stem_sentence.append(stem)

    text = [w for w in stem_sentence if w not in stopwords_list]
    text = ' '.join(text)

    return text


classifier = None
vectorizer = None

def train(df):
    global classifier, vectorizer
    print("Training...")
    sentences = df["description"].values
    vectorizer = CountVectorizer(min_df=0, lowercase=False)
    vectorizer.fit(sentences)
    vectorizer.transform(sentences)

    X = df.drop("Unnamed: 0", axis=1)
    X = X.drop("genre", axis=1).drop("id", axis=1).drop("title", axis=1)
    Y = df["genre"].values


    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, Y, test_size=1/3)#, random_state=50
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)

    X_train = vectorizer.transform(sentences_train)
    X_test  = vectorizer.transform(sentences_test)

    # classifier = LogisticRegression().fit(X_train, y_train)
    # # predicted = classifier.predict(X_test)
    # # y_err = Y - predicted
    # linear_regression = LinearRegression().fit(X_train, y_train)

    # X_ensemble = pd.DataFrame(
    #     {
    #         "LOGISTIC": classifier.predict(X_test),
    #         "LINEAR": linear_regression.predict(X_test)
    #     }
    # )
    # classifier = LinearRegression().fit(X_ensemble, y_train)

    # classifier = MLPClassifier(hidden_layer_sizes=(13,13,13), max_iter=300)
    # classifier.fit(X_train, y_train)

    # classifier = CategoricalNB().fit(X_train, y_train)
    classifier = ComplementNB()
    classifier.fit(X_train, y_train)
    predict = classifier.predict(X_test)
    score = classifier.score(X_test, y_test)
    print(f"Trained with {score:.3f}% accuracy")
    metrics.accuracy_score(predict, y_test)

    return score



def accuracy(df):
    genre = df["genre"]
    genre_predicted = df["genre_predit"]
    length_genre = len(genre)
    error = 0
    for i in range(length_genre):
        if genre[i] != genre_predicted[i]:
            error += 1

    print(f"Error : {(error/length_genre)*100:.3f}%")
    


def predict(file_path, output, df_trained, df_to_predict):
    global classifier, vectorizer
    if classifier is None:
        raise Exception("Not trained yet")

    # df_to_predict = pd.read_csv(file_path)

    X_test_to_predict = vectorizer.transform(df_to_predict["description"])



    classified = classifier.predict(X_test_to_predict)

    # df2 = pd.DataFrame(data={"title": df_to_predict["title"].values, "genre": df_trained["genre"], "genre_predit": classified})
    df2 = pd.DataFrame(data={"title": df_to_predict["title"].values, "genre": df_to_predict["genre"], "genre_predit": classified})

    accuracy(df2)
    df2.to_csv(output)

def convert_string_to_dataset_prediction(title, description, genre_attendu):
    return pd.DataFrame({"title": title, "description": clean_text(description), "genre": genre_attendu}, index=[0])


def keras(df):
    sentences = df["description"].values

    encoder = LabelEncoder()
    sentences_labels = encoder.fit_transform(sentences)
    encoder = OneHotEncoder(sparse=False)
    sentences_labels = sentences_labels.reshape((54200, 1))
    encoder.fit_transform(sentences_labels)

    Y = df["genre"].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences_labels, Y, test_size=1/3)#, random_state=50

    
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences_train)

    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    print(sentences_train[2])
    print(X_train[2])

    maxlen = 100

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    print(X_train[0, :])

    embedding_dim = 50

    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, 
                            output_dim=embedding_dim, 
                            input_length=maxlen))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    model.summary()
    


if __name__ == "__main__":
    df_trained_data = pd.read_csv("archive/dataset_csv/test_data_solution_clean.csv")
    print(df_trained_data["genre"].unique())
    #keras(df_trained_data)
    df=convert_string_to_dataset_prediction("les tuches 4", "Twenty-five years after the original series of murders in Woodsboro, a new killer emerges, and Sidney Prescott must return to uncover the truth.", "horror")
    score = train(df_trained_data)
    predict("archive/dataset_csv/test_data.csv", "predicted.csv", df_trained_data, df)
    print("Accuracy:", score)


# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)


# X_train_scaling = StandardScaler().fit(X_train)
# X_test_scaling = StandardScaler().fit(X_test)

# X_train = X_train_scaling.transform(X_train)
# X_test = X_test_scaling.transform(X_test)


# MLPC = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)

# trained = MLPC.fit(X_train, Y_train)
# predictions = trained.predict(X_test)


# M = confusion_matrix(Y_test, predictions)

# print(M)


"""
on a essayer d'utiliser MLPClassifier mais il fallait que des nombre et l description est pas un nombre donc on a du la vectoriser
"""