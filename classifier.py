from os import error
from keras.backend import softmax
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
from sklearn.svm import LinearSVC, SVC
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
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import tensorflow as tf

# Import necessary modules

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt


# Keras specific

import keras

from keras.models import Sequential

from keras.layers import Dense

from tensorflow.keras.utils import to_categorical


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

def train(df, test_size, train_size):
    global classifier, vectorizer
    print("Training...")
    """sentences = df["description"].values
    vectorizer = CountVectorizer(min_df=0, lowercase=False)
    vectorizer.fit(sentences)
    vectorizer.transform(sentences)

    X = df.drop("Unnamed: 0", axis=1)
    X = X.drop("genre", axis=1).drop("id", axis=1).drop("title", axis=1)
    Y = df["genre"].values"""

    sentences = shuffle(df["description"].values, n_samples=5000)
    vectorizer = CountVectorizer(min_df=0, lowercase=False)
    vectorizer.fit(sentences)
    vectorizer.transform(sentences)

    X = df.drop("Unnamed: 0", axis=1).drop("genre", axis=1).drop("id", axis=1).drop("title", axis=1)
    X = shuffle(X, n_samples=5000)
    Y = shuffle(df["genre"], n_samples=5000)


    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, Y, test_size=test_size, train_size=train_size)#, random_state=50
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)

    X_train = vectorizer.transform(sentences_train)
    X_test  = vectorizer.transform(sentences_test)

    # classifier = LinearRegression().fit(X_ensemble, y_train)

    # classifier = MLPClassifier(hidden_layer_sizes=(13,13,13), max_iter=300)
    # classifier.fit(X_train, y_train)

    # classifier = CategoricalNB().fit(X_train, y_train)

    # classifier = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam")
    classifier = SVC()
    classifier.fit(X_train, y_train)
    predict = classifier.predict(X_test)
    score = classifier.score(X_test, y_test)
    print(f"Trained with {score:.3f}% accuracy")
    metrics.accuracy_score(predict, y_test)

    return score



def predictError(df):
    genre = df["genre"]
    genre_predicted = df["genre_predit"]
    length_genre = len(genre)
    error = 0
    for i in range(length_genre):
        if genre[i] != genre_predicted[i]:
            error += 1

    print(f"Error : {(error/length_genre)*100:.3f}%")
    return (error/length_genre)*100



def predict(file_path, output, df_trained, df_to_predict):
    global classifier, vectorizer
    if classifier is None:
        raise Exception("Not trained yet")

    # df_to_predict = pd.read_csv(file_path)
    print("Predicting...")
    X_test_to_predict = vectorizer.transform(df_to_predict["description"])



    classified = classifier.predict(X_test_to_predict)

    # df2 = pd.DataFrame(data={"title": df_to_predict["title"].values, "genre": df_trained["genre"], "genre_predit": classified})
    df2 = pd.DataFrame(data={"title": df_to_predict["title"].values, "genre": df_to_predict["genre"], "genre_predit": classified})

    predictError(df2)
    df2.to_csv(output)

def convert_string_to_dataset_prediction(title, description, genre_attendu):
    return pd.DataFrame({"title": title, "description": clean_text(description), "genre": genre_attendu}, index=[0])

def create_model(optimizer='rmsprop', init='glorot_uniform'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu'))
	model.add(Dense(8, kernel_initializer=init, activation='relu'))
	model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

def keras(df):
    from sklearn.preprocessing import StandardScaler
    from keras.models import Sequential
    from keras.layers import Dense


    sentences = df["description"].values

    Y = np.ravel(df["genre"].values)

    X_train, X_test, y_train, y_test = train_test_split(sentences, Y, test_size=0.30, random_state=40)


    scaler = CountVectorizer().fit(X_train)

    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)
    print(X_train.dtype)

    model = Sequential()

    # model.add(Dense(8, activation='relu', input_shape=(8,)))

    # model.add(Dense(8, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

    model.fit(X_train, y_train,epochs=4, batch_size=1, verbose=1)
    for layer in model.layers:
        weights = layer.get_weights()
    from keras.utils import plot_model
    plot_model(model, to_file='/tmp/model.png', show_shapes=True,)
    y_pred = model.predict_classes(X_test)

    score = model.evaluate(X_test, y_test,verbose=1)

    print(score)





    # y_train = to_categorical(np.asarray(pd.factorize(y_train)[0]))

    # y_test = to_categorical(np.asarray(pd.factorize(y_test)[0]))

    # # y_train = to_categorical(y_train, 2)
    # # y_test = to_categorical(y_test, 2)


    # count_classes = y_test.shape[1]
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(1)
    #     ])

    # # model.add(Dense(500, activation='relu', input_dim=8))

    # # model.add(Dense(100, activation='relu'))

    # # model.add(Dense(50, activation='relu'))

    # model.add(Dense(27, activation='softmax'))


    # # Compile the model

    # # model.compile(optimizer='adam',

    # #             loss='binary_crossentropy',
    # #             softmax="sigmoid",

    # #             metrics=['accuracy'])

    # model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
    #           optimizer=tf.keras.optimizers.SGD(), # SGD is short for stochastic gradient descent
    #           metrics=["mae"])

    # model.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=20)
    # pred_train= model.predict(X_train)

    # scores = model.evaluate(X_train, y_train, verbose=0)

    # print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))



    # pred_test= model.predict(X_test)

    # scores2 = model.evaluate(X_test, y_test, verbose=0)

    # print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))



def graphAccuracy(df_trained_data, df_to_predict):
    test_sizes=[x/10 for x in range(1, 10)]
    train_sizes=[x/10 for x in range(1, 10)]

    x_labels = [0]
    x_ticks=[0]
    i=0
    for a in test_sizes:
        for b in train_sizes:
            x_labels.append("("+str(a)+","+str(b)+")")
            i+=1
            x_ticks.append(i)
    i=1
    plt.xticks(ticks=x_ticks, labels=x_labels)
    x = []
    y = []

    nbOfGraph=0
    for test_size in test_sizes:
        for train_size in train_sizes:
            if test_size+train_size<1:
                print(str(test_size)+" "+str(train_size))
                print(x_ticks[i])

                score = train(df_trained_data, test_size, train_size)
                y.append(score)
                x.append(x_ticks[i])
                predict("archive/dataset_csv/test_data.csv", "predicted.csv", df_trained_data, df_to_predict)
                print("Accuracy:", score)

            else:
                y.append(0)
                x.append(x_ticks[i])

            if i!=0 and i%9 ==0:
                    plt.scatter(x, y)

                    plt.savefig("variation_accuracy"+str(nbOfGraph)+".png")
                    x = []
                    y = []
                    plt.clf()
                    plt.xticks(ticks=x_ticks, labels=x_labels)

                    nbOfGraph+=1
            i+=1


    #plt.scatter(x, y)

    #plt.savefig("variation_accuracy.png")

def graphError(df_trained_data, df_to_predict):
    test_sizes=[x/10 for x in range(1, 10)]
    train_sizes=[x/10 for x in range(1, 10)]

    x_labels = [0]
    x_ticks=[0]
    i=0
    for a in test_sizes:
        for b in train_sizes:
            x_labels.append("("+str(a)+","+str(b)+")")
            i+=1
            x_ticks.append(i)
    i=1
    plt.xticks(ticks=x_ticks, labels=x_labels)
    x = []
    y = []

    nbOfGraph=0
    for test_size in test_sizes:
        for train_size in train_sizes:
            if test_size+train_size<1:
                print(str(test_size)+" "+str(train_size))
                print(x_ticks[i])

                score = train(df_trained_data, test_size, train_size)
                predict("archive/dataset_csv/test_data.csv", "predicted.csv", df_trained_data, df_to_predict)
                error=predictError(pd.read_csv("predicted.csv"))

                y.append(error)
                x.append(x_ticks[i])
                print("Accuracy:", score)

            else:
                y.append(0)
                x.append(x_ticks[i])

            if i!=0 and i%9 ==0:
                    plt.scatter(x, y)

                    plt.savefig("variation_error"+str(nbOfGraph)+".png")
                    x = []
                    y = []
                    plt.clf()
                    plt.xticks(ticks=x_ticks, labels=x_labels)

                    nbOfGraph+=1
            i+=1


    #plt.scatter(x, y)

    #plt.savefig("variation_accuracy.png")



if __name__ == "__main__":
    df_trained_data = pd.read_csv("archive/dataset_csv/train_data_clean.csv")
    # print(df_trained_data["genre"].unique())
    keras(df_trained_data)
    #df=convert_string_to_dataset_prediction("les tuches 4", "Twenty-five years after the original series of murders in Woodsboro, a new killer emerges, and Sidney Prescott must return to uncover the truth.", "horror")
    #df_to_predict = pd.read_csv("archive/dataset_csv/test_data_solution_clean.csv")
    #train(df_trained_data, 0.1, 0.1)
    # predict("archive/dataset_csv/test_data.csv", "predicted.csv", df_trained_data, df_to_predict)
    # keras(df_trained_data)
    # graphError(df_trained_data, df_to_predict)
    #score = train(df_trained_data, 2/3, 1/3)


    #print("Accuracy:", score)


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