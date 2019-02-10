from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from time import time


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

    print("F1 score dla zbioru trenującego: {:.4f}.".format(predict_labels(clf, x_train, y_train)))
    print("F1 score dla zbioru testującego: {:.4f}.".format(predict_labels(clf, x_test, y_test)))


def performance_metric(y_true, y_predict):
    error = f1_score(y_true, y_predict, pos_label=1)
    return error


def fit_model(classifier, x, y):

    parameters = {'kernel': ['poly', 'rbf', 'sigmoid'], 'degree': [1, 2, 3], 'C': [0.1, 1, 10]}

    f1_scorer = make_scorer(performance_metric,
                            greater_is_better=True)

    clf = GridSearchCV(classifier,
                       param_grid=parameters,
                       scoring=f1_scorer)

    clf.fit(x, y)

    return clf

def read_data(path):


    return clf
