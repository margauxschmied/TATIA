import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

df = pd.read_csv("archive/dataset_csv/train_data_clean.csv")

sentences = df["description"].values
vectorizer = TfidfVectorizer()
vectorizer.fit(sentences)
vectorizer.transform(sentences)

Y = df["genre"].values

X_train, X_test, y_train, y_test = train_test_split(
    sentences, Y, test_size=0.1, train_size=0.5)
vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', MultinomialNB()))
models.append(('SVM', LinearSVC()))
models.append(('MLP', MLPClassifier(hidden_layer_sizes=(10,))))

results = []
names = []
scoring = 'accuracy'
for name, model in models:
    print(f"Testing with {name}...")
    kfold = model_selection.KFold(n_splits=8)
    cv_results = model_selection.cross_val_score(
        model, X_train, y_train, scoring=scoring, n_jobs=-1, verbose=1, cv=kfold) * 100
    mean = cv_results.mean()
    results.append(mean)
    names.append(name)
    msg = "%s: %f (%f)" % (name, mean, cv_results.std())
    print(msg)

plt.title('How to compare sklearn classification algorithms')
plt.bar(names, results, width=0.7, color=plt.get_cmap(
    'Paired').colors, edgecolor='k', linewidth=2)
plt.xlabel("Classifier")
plt.ylabel("Accuracy (%)")
plt.savefig("compare_classifier.png")
