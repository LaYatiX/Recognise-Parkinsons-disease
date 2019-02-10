import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

import functions

clf1 = GaussianNB() #Naive Bayes
clf2 = svm.SVC(kernel="linear", tol=1e-3, gamma="scale") #Support Vector Machines
clf3 = SGDClassifier(loss="epsilon_insensitive") #Stochastic Gradient Descent
clf4 = RidgeClassifier(solver="cholesky") #Ridge Classifier
clf5 = GradientBoostingClassifier(n_estimators=80, learning_rate=1.0, max_depth=1, random_state=0) #Gradient Boosting Classifier
clf6 = AdaBoostClassifier(n_estimators=80, learning_rate=1.1) #Ada Boost Classifier

parkinson_csv = pd.read_csv("parkinsons.csv")
print("Dane pacjentów zostały wczytane!")

sample = parkinson_csv.shape[0]

features = parkinson_csv.shape[1] - 1

n_parkinsons = parkinson_csv[parkinson_csv['status'] == 1].shape[0]

n_healthy = parkinson_csv[parkinson_csv['status'] == 0].shape[0]

print("Liczba próbek: {}".format(sample))
print("Liczba cech: {}".format(features))
print("Liczba próbek z Parkinsonem: {}".format(n_parkinsons))
print("Liczba próbek bez Parkinsona: {}".format(n_healthy))

feature_cols = list(parkinson_csv.columns[1:16]) + list(parkinson_csv.columns[18:])
target_col = parkinson_csv.columns[17]

print("Cechy:\n{}".format(feature_cols))

x_all = parkinson_csv[feature_cols]
y_all = parkinson_csv[target_col]

print("\nWartości cech:")
print(x_all.head())

num_all = parkinson_csv.shape[0]
num_train = 150
num_test = num_all - num_train


x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test, random_state=5)

print("Zbiór trenujący: {} samples".format(x_train.shape[0]))
print("Zbiór testujący: {} samples".format(x_test.shape[0]))


x_train_all = x_train[:150]
y_train_all = y_train[:150]

select = 0
clf = 0
while True:
    print('')
    print('MENU')
    print("1.Klasyfikacja Naive Bayes")
    print("2.Klasyfikacja Support Vector Machines")
    print("3.Klasyfikacja Stochastic Gradient Descent")
    print("4.Klasyfikacja Ridge Classifier")
    print("5.Klasyfikacja Gradient Boosting Classifier")
    print("6.Klasyfikacja Ada Boost Classifier")
    print("Wybierz 0 by wyjść")
    select = input('Podaj opcje:')
    print('')
    if select == '1':
        print("Wybrana Klasyfikacja: Naive Bayes")
        functions.train_predict(clf1, x_train_all, y_train_all, x_test, y_test)
        clf = clf1
    elif select == '2':
        print("Wybrana Klasyfikacja: Support Vector Machines")
        functions.train_predict(clf2, x_train_all, y_train_all, x_test, y_test)
        clf = clf2
    elif select == '3':
        print("Wybrana Klasyfikacja: Stochastic Gradient Descent")
        functions.train_predict(clf3, x_train_all, y_train_all, x_test, y_test)
        clf = clf3
    elif select == '4':
        print("Wybrana Klasyfikacja: Ridge Classifier")
        functions.train_predict(clf4, x_train_all, y_train_all, x_test, y_test)
        clf = clf4
    elif select == '5':
        print("Wybrana Klasyfikacja: Gradient Boosting Classifier")
        functions.train_predict(clf5, x_train_all, y_train_all, x_test, y_test)
        clf = clf5
    elif select == '6':
        print("Wybrana Klasyfikacja: Gradient Boosting Classifier")
        functions.train_predict(clf6, x_train_all, y_train_all, x_test, y_test)
        clf = clf6

    elif select == '0':
        break
    else:
        print("Niepoprawny kod. Spróbuj ponownie.")

    if select > '0' and select <= '6':
        score = 'precision'
        print('Trenowanie modelu................................................')
        clf_out = functions.fit_model(clf, x_train, y_train)
        print("Dopasowanie modelu zakończone!")

        print("Najlepsze wartości to: ")

        print(clf_out.best_params_)

        print("Score modelu dla modelu trenującego: {:.4f}.".format(functions.predict_labels(clf_out, x_train, y_train)))
        print("Score modelu dla modelu testująecgo: {:.4f}.".format(functions.predict_labels(clf_out, x_test, y_test)))

        print("Trenowanie modelu zakończone")

