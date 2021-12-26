import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
from sklearn.utils import shuffle


df = pd.read_csv("archive/dataset_csv/train_data_clean.csv")

sentences = df["description"].values
vectorizer = CountVectorizer()
vectorizer.fit(sentences)
vectorizer.transform(sentences)

Y = df["genre"].values


X_train, X_test, y_train, y_test = train_test_split(sentences, Y, test_size=0.2, train_size=0.7)
vectorizer = CountVectorizer()
vectorizer.fit(X_train)

X_train = vectorizer.transform(X_train)
X_test  = vectorizer.transform(X_test)


kfold = model_selection.KFold(n_splits=10)




models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', MultinomialNB()))
models.append(('SVM', LinearSVC()))
models.append(('MLP', MLPClassifier()))


results = []
names = []
scoring = 'accuracy'
for name, model in models:
    print(f"Testing with {name}...")
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring) * 100
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure(figsize=(10,10))
fig.suptitle('How to compare sklearn classification algorithms')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
ax.set_xlabel("Classifier")
ax.set_ylabel("Accuracy (%)")
plt.savefig("compare_classifier.png")