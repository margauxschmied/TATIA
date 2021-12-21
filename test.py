import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, NuSVC, SVR, NuSVR, OneClassSVM, LinearSVC, LinearSVR
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier, MLPRegressor

import pandas as pd
from sklearn.utils import shuffle


df = pd.read_csv("archive/dataset_csv/train_data_clean.csv")

sentences = shuffle(df["description"].values, n_samples=2000)
vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences)
vectorizer.transform(sentences)

X = df.drop("Unnamed: 0", axis=1).drop("genre", axis=1).drop("id", axis=1).drop("title", axis=1)
X = shuffle(X, n_samples=2000)
Y = shuffle(df["genre"], n_samples=2000)

print(len(X), len(Y))
X_train, X_test, y_train, y_test = train_test_split(sentences, Y, test_size=1/3)#, random_state=50
vectorizer = CountVectorizer()
vectorizer.fit(X_train)

X_train = vectorizer.transform(X_train)
X_test  = vectorizer.transform(X_test)


# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
kfold = model_selection.KFold(n_splits=10)




models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('MNB', MultinomialNB()))
models.append(('SVC', SVC()))
models.append(('LSVC', LinearSVC()))
models.append(('MLPC', MLPClassifier()))


results = []
names = []
scoring = 'accuracy'
for name, model in models:
    print("Studying ", name, "...")
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure(figsize=(10,10))
fig.suptitle('How to compare sklearn classification algorithms')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.savefig("compare_classifier_working.png")
# plt.show()