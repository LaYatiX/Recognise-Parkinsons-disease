from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from time import time
from sklearn import svm


def train_classifier(clf, x_train, y_train):

    start = time()
    clf.fit(x_train, y_train)
    end = time()

    print("Czas trenowania modelu:  {:.4f} sekund".format(end - start))


def predict_labels(clf, features, target):

    start = time()
    y_pred = clf.predict(features)
    end = time()

    print("Czas przewidywania: {:.4f} sekund.".format(end - start))
    return f1_score(target.values, y_pred, pos_label=1)


def train_predict(clf, x_train, y_train, x_test, y_test):

    print("Trenowanie {} przy użyciu zbioru trenującego o rozmiarze {}. . .".format(clf.__class__.__name__, len(x_train)))

    train_classifier(clf, x_train, y_train)

    print("Score modelu dla modelu trenującego: {:.4f}.".format(predict_labels(clf, x_train, y_train)))
    print("Score modelu dla modelu testującego: {:.4f}.".format(predict_labels(clf, x_test, y_test)))


def performance_metric(y_true, y_predict):
    error = f1_score(y_true, y_predict, pos_label=1)
    return error


def fit_model(classifier, x, y):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    score = 'precision'
    print("Szukanie najlepszych wartosci parametrów")
    clf = GridSearchCV(svm.SVC(kernel="linear", tol=1e-3, gamma="scale"), tuned_parameters, cv=5, scoring='%s_macro' % score)
    clf.fit(x, y)

    return clf
