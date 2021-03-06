import argparse
import pickle
import re
import string
import sys
import warnings

import matplotlib.pyplot as plt
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")

stopwords_list = stopwords.words('english')
stopwords_list += list(string.punctuation)
stopwords_list += ['one', 'two', 'three', 'four', 'five', 'six', 'seven',
                   'eight', 'nine', 'go', 'goes', 'get', 'also', 'however', 'tells']
stopwords_list += [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
stopwords_list += [x for x in string.ascii_lowercase]


def print_genre():
    df_path = "archive/dataset_csv/train_data_clean.csv"
    df = pd.read_csv(df_path)
    genres = []
    for i in range(len(df["genre"])):
        if df["genre"][i] not in genres:
            genres.append(df["genre"][i])
    print(genres)


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


def save_model(model, file_path):
    pickle.dump(model, open(file_path, "wb+"))
    print(f"{file_path} saved")


def load_model(file_path):
    model = pickle.load(open(file_path, "rb"))
    print(f"{model} loaded")
    print(model["vectorizer"])
    return model


class TrainClassifier(object):
    def __init__(self, df_path="archive/dataset_csv/train_data_clean.csv", test_size=0.1, train_size=0.5,
                 sampling=None):
        self.vectorizer = None
        self.df = pd.read_csv(df_path)
        self.classifier = None
        self.vectorizer = TfidfVectorizer()
        self.test_size = test_size
        self.train_size = train_size
        self.sentences = self.df["description"].values
        self.Y = self.df["genre"].values
        self.sampling = sampling if sampling is not None else None

    def train(self, classifier, model_path="neural_network_model.sav", **kwargs):
        print(f"Start training with {classifier}...")

        if self.sampling is not None:
            self.sentences = shuffle(self.sentences, n_samples=self.sampling)
            self.Y = shuffle(self.Y, n_samples=self.sampling)

        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(self.sentences)
        self.vectorizer.transform(self.sentences)

        sentences_train, sentences_test, y_train, y_test = train_test_split(
            self.sentences, self.Y, test_size=self.test_size, train_size=self.train_size)  # , random_state=50
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(sentences_train)

        X_train = self.vectorizer.transform(sentences_train)
        X_test = self.vectorizer.transform(sentences_test)

        self.classifier = classifier(**kwargs)

        self.classifier.fit(X_train, y_train)

        save_model(dict(classifier=self.classifier, vectorizer=self.vectorizer),
                   model_path)

        score = self.classifier.score(X_test, y_test) * 100
        print(f"Trained with {score:.3f}% accuracy")
        metrics.ConfusionMatrixDisplay.from_estimator(
            self.classifier, X_test, y_test, labels=self.df["genre"].unique(), include_values=False, xticks_rotation=90)
        plt.savefig(f'confusion_matrix_{self.classifier}.png', dpi=400)
        plt.clf()
        return score


def graph_train_naif_logistic_regression(df_trained_data_path, df_to_predict):
    test_sizes = [x / 10 for x in range(1, 10)]
    train_sizes = [x / 10 for x in range(1, 10)]

    x_labels = [0]
    x_ticks = [0]
    i = 0
    for a in test_sizes:
        for b in train_sizes:
            x_labels.append("(" + str(a) + "," + str(b) + ")")
            i += 1
            x_ticks.append(i)
    i = 1

    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=10)
    plt.xlabel("(test_size, train_size)")
    plt.ylabel("Percentage (%)")
    plt.title("Na??f Logistic Regression variation")
    plt.grid(True)
    x = []
    y_accuracy = []
    y_error = []

    nbOfGraph = 0
    for test_size in test_sizes:
        for train_size in train_sizes:
            if test_size + train_size < 1:
                print(str(test_size) + " " + str(train_size))
                print(x_ticks[i])
                print(x_labels[i])

                trainer = TrainClassifier(
                    df_trained_data_path, test_size, train_size)
                score = trainer.train(
                    classifier=LogisticRegression, model_path="naif_logistic_regression_model.sav")

                predict("predicted_logistic_regression.csv",
                        df_to_predict, classifier_method="logistic_regression")
                error = predictError(pd.read_csv(
                    "predicted_logistic_regression.csv"))
                y_error.append(error)

                y_accuracy.append(score * 100)
                x.append(x_ticks[i])
                print("Accuracy:", score)

            if i != 0 and i % 9 == 0:
                plt.scatter(x, y_accuracy)
                plt.scatter(x, y_error)
                plt.legend(["Accuracy", "Error"])

                plt.savefig(
                    f"variation-error_accuracy-{nbOfGraph}.png")
                y_accuracy = []
                y_error = []
                x = []
                plt.clf()
                plt.xticks(ticks=x_ticks, labels=x_labels, rotation=10)
                plt.xlabel("(test_size, train_size)")
                plt.ylabel("Percentage (%)")
                plt.title("Na??f Logistic Regression variation")
                plt.grid(True)

                nbOfGraph += 1
            i += 1


def graph_train_svc(df_trained_data_path, df_to_predict):
    test_sizes = [x / 10 for x in range(1, 10)]
    train_sizes = [x / 10 for x in range(1, 10)]

    x_labels = [0]
    x_ticks = [0]
    i = 0
    for a in test_sizes:
        for b in train_sizes:
            x_labels.append("(" + str(a) + "," + str(b) + ")")
            i += 1
            x_ticks.append(i)
    i = 1

    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=10)
    plt.xlabel("(test_size, train_size)")
    plt.ylabel("Percentage (%)")
    plt.title("SVM variation")
    plt.grid(True)
    x = []
    y_accuracy = []
    y_error = []

    nbOfGraph = 0
    for test_size in test_sizes:
        for train_size in train_sizes:
            if test_size + train_size < 1:
                print(str(test_size) + " " + str(train_size))
                print(x_ticks[i])
                print(x_labels[i])

                trainer = TrainClassifier(
                    df_trained_data_path, test_size, train_size)
                score = trainer.train(
                    classifier=LinearSVC, model_path="svm_model.sav")

                predict("predicted_svm.csv",
                        df_to_predict, classifier_method="svm")
                error = predictError(pd.read_csv(
                    "predicted_svm.csv"))
                y_error.append(error)

                y_accuracy.append(score * 100)
                x.append(x_ticks[i])
                print("Accuracy:", score)

            if i != 0 and i % 9 == 0:
                plt.scatter(x, y_accuracy)
                plt.scatter(x, y_error)
                plt.legend(["Accuracy", "Error"])

                plt.savefig(
                    f"svm-variation-{nbOfGraph}.png")
                y_accuracy = []
                y_error = []
                x = []
                plt.clf()
                plt.xticks(ticks=x_ticks, labels=x_labels, rotation=10)
                plt.xlabel("(test_size, train_size)")
                plt.ylabel("Percentage (%)")
                plt.title("SVM variation")
                plt.grid(True)

                nbOfGraph += 1
            i += 1


def graph_train_neural_network(df_trained_data_path, df_to_predict):
    x_labels = [0]
    x_ticks = [0]
    i = 0
    for layer in range(1, 5):
        for neuron in range(1, 101, 9):
            x_labels.append(str(layer) + " x " + str(neuron))
            i += 1
            x_ticks.append(i)
    i = 1

    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=10)
    plt.xlabel("layer X neurons")
    plt.ylabel("Percentage (%)")
    plt.title("Neural Network MLPClassifier variation")
    plt.grid(True)

    x = []
    y_accuracy = []
    y_error = []

    nbOfGraph = 0
    for layer in range(1, 5):
        for neuron in range(1, 101, 9):
            print(str(layer) + " x " + str(neuron))
            print(x_ticks[i])
            print(x_labels[i])
            trainer = TrainClassifier(
                df_trained_data_path, 0.2, 0.7)
            score = trainer.train(classifier=MLPClassifier,
                                  hidden_layer_sizes=tuple((neuron for i in range(layer))))

            predict("predicted_neural_network.csv",
                    df_to_predict, classifier_method="neural_network")
            error = predictError(pd.read_csv(
                "predicted_neural_network.csv"))

            y_error.append(error)
            y_accuracy.append(score * 100)
            x.append(x_ticks[i])

            print("Accuracy:", score)
            i += 1

        plt.scatter(x, y_accuracy)
        plt.scatter(x, y_error)
        plt.legend(["Accuracy", "Error"])
        plt.savefig(
            f"neural_network_variation-error_accuracy-{nbOfGraph}.png")
        y_accuracy = []
        y_error = []
        x = []
        plt.clf()

        plt.xticks(ticks=x_ticks, labels=x_labels, rotation=10)
        plt.xlabel("layer X neurons")
        plt.ylabel("Percentage (%)")
        plt.title("Neural Network MLPClassifier variation")
        plt.grid(True)

        nbOfGraph += 1


def predictError(df):
    genre = df["genre"]
    genre_predicted = df["genre_predit"]
    length_genre = len(genre)
    error = 0
    for i in range(length_genre):
        if genre[i] != genre_predicted[i]:
            error += 1

    print(f"Error : {(error / length_genre) * 100:.3f}%")
    return (error / length_genre) * 100


def predict(output, df_to_predict, classifier_method="neural_network", sampling=None,
            model_path="neural_network_model.sav", **kwargs):
    try:
        model = load_model(model_path)
    except:
        trainer = TrainClassifier(sampling=sampling)
        if classifier_method == "neural_network":
            trainer.train(MLPClassifier, **kwargs)
        elif classifier_method == "logistic_regression":
            trainer.train(LogisticRegression, **kwargs)
        elif classifier_method == "svm":
            trainer.train(LinearSVC, **kwargs)

        model = load_model(model_path)

    classifier = model["classifier"]
    vectorizer = model["vectorizer"]
    print(f"Predicting with {classifier}...")

    X_test_to_predict = vectorizer.transform(df_to_predict["description"])

    classified = classifier.predict(X_test_to_predict)
    try:
        # calcul du pourcentage d'erreurs
        df2 = pd.DataFrame(data={
            "title": df_to_predict["title"].values, "description": df_to_predict["description"],
            "genre": df_to_predict["genre"], "genre_predit": classified})
    except:
        df2 = pd.DataFrame(data={
            "title": df_to_predict["title"].values, "description": df_to_predict["description"],
            "genre_predit": classified})

    # predictError(df2)
    df2.to_csv(output)
    print(f"Prediction done and saved in {output}")
    print(f"Classifier {classifier} prediction for {df_to_predict['title'].values} is {classified}")


def convert_string_to_dataset_prediction(title, description):
    return pd.DataFrame({"title": title, "description": clean_text(description)}, index=[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict genre of a movie')
    parser.add_argument('-t', '--title', type=str,
                        help="Title of the movie", dest="title", default="")
    parser.add_argument('-d', '--description', type=str,
                        help="Description of the movie", dest="description", default=None)

    parser.add_argument('-cl', '--classifier', type=str,
                        help="Classifier to use", dest="classifier", default="neural_network")
    parser.add_argument('-mp', '--model_path', type=str, help="Path to the model",
                        dest="model_path", default="neural_network_model.sav")
    parser.add_argument('-tr', '--train', type=bool,
                        help="Train the model", dest="train", default=False)
    parser.add_argument('-csv', '--csv_output', type=str,
                        help="CSV output file", dest="csv_output", default="predicted.csv")
    parser.add_argument('-n', '--neurons', type=int,
                        help="Number of neurons in the hidden layer", dest="neurons", default=100)
    parser.add_argument('-l', '--layers', type=int,
                        help="Number of hidden layers", dest="layers", default=1)

    parser.add_argument('-trs', '--train_size', type=float,
                        help="Train size", dest="train_size", default=0.5)
    parser.add_argument('-tes', '--test_size', type=float,
                        help="Test size", dest="test_size", default=0.1)
    parser.add_argument('-csv_train', '--csv_train', type=str, help="CSV train file", dest="csv_train",
                        default="archive/dataset_csv/train_data_clean.csv")
    args = parser.parse_args()
    print(args)

    if args.train:
        trainer = TrainClassifier(
            df_path=args.csv_train,
            test_size=args.test_size, train_size=args.train_size)
        if args.classifier == "neural_network":
            trainer.train(
                classifier=MLPClassifier,
                hidden_layer_sizes=tuple(
                    (args.neurons for i in range(args.layers))),
                model_path=args.model_path)
        elif args.classifier == "logistic_regression":
            trainer.train(classifier=LogisticRegression,
                          model_path=args.model_path)
        elif args.classifier == "svm":
            trainer.train(classifier=LinearSVC, model_path=args.model_path)
        else:
            print("Unknown classifier see python3 classifier.py -h")
            sys.exit(1)

    if args.description is not None:
        predict(args.csv_output, convert_string_to_dataset_prediction(
            args.title, args.description), classifier_method=args.classifier, model_path=args.model_path)

    sys.exit(0)
