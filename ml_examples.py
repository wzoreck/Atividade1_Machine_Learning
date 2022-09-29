#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import pandas as pandas
import numpy as numpy
import seaborn as seaborn
import matplotlib.pyplot as pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


base_car_evaluation = pandas.read_csv('car_data.csv')

base_car_evaluation

## Visualização dos dados
base_car_evaluation.describe()

seaborn.countplot(x=base_car_evaluation['buying'])
seaborn.countplot(x=base_car_evaluation['maint'])
seaborn.countplot(x=base_car_evaluation['doors'])
seaborn.countplot(x=base_car_evaluation['persons'])
seaborn.countplot(x=base_car_evaluation['lug_boot'])
seaborn.countplot(x=base_car_evaluation['safety'])

seaborn.countplot(x=base_car_evaluation['class'])

# Visualizar a distribuição dos dados por atributo
numpy.unique(base_car_evaluation['buying'], return_counts=True)
numpy.unique(base_car_evaluation['maint'], return_counts=True)
numpy.unique(base_car_evaluation['doors'], return_counts=True)
numpy.unique(base_car_evaluation['persons'], return_counts=True)
numpy.unique(base_car_evaluation['lug_boot'], return_counts=True)
numpy.unique(base_car_evaluation['safety'], return_counts=True)

numpy.unique(base_car_evaluation['class'], return_counts=True)

base_car_evaluation.isnull().sum()

## Divisão da base (Previsores e Classes)
X_car_attributes = base_car_evaluation.iloc[:, 0:6].values
X_car_attributes

y_car_class = base_car_evaluation.iloc[:, 6].values
y_car_class

## Normalização dos Previsores
# Tratamento de atributos categóricos, LABELENCODER - converte string para inteiro ex: P M G -> 0 1 2
label_encoder = LabelEncoder()

index = [0, 1, 2, 3, 4, 5] # Vai ser usado em todos os atributos por conta dos valores (5-more e more) nos atributos que "são numéricos"

for i in index:
    X_car_attributes[:, i] = label_encoder.fit_transform(X_car_attributes[:, i])

X_car_attributes

# ONEHOTENCODER - Não há peso por ser apenas 0 e 1 (há algoritmos que isso interfere)
'''
onehotencoder_car_evaluation = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), index)], remainder='passthrough')

X_car_attributes = onehotencoder_car_evaluation.fit_transform(X_car_attributes).toarray()

X_car_attributes.shape
'''

## Escalonamento dos valores (Transformar a base de dados em uma "medida padrão")
scaler_car_evaluation = StandardScaler()

X_car_attributes = scaler_car_evaluation.fit_transform(X_car_attributes)

## Divisão base Treinamento e Teste
X_car_attributes_training, X_car_attributes_test, y_car_class_training, y_car_class_test = train_test_split(X_car_attributes, y_car_class, test_size=0.25, random_state=0)

# Salvar as variáveis prontas para uso
with open('car_evaluation_attributes_and_class.pkl', mode='wb') as f:
    pickle.dump([X_car_attributes_training, X_car_attributes_test, y_car_class_training, y_car_class_test], f)


################################### Algorítmos de ML ###################################

## Buscar as variáveis prontas para treinamento e teste dos algorítmos
with open('car_evaluation_attributes_and_class.pkl', mode='rb') as f:
    X_car_attributes_training, X_car_attributes_test, y_car_class_training, y_car_class_test = pickle.load(f)


########### 01 - Naive Bayes ###########
from sklearn.naive_bayes import GaussianNB

naive_bayes_car_evaluation = GaussianNB()

## Aprendizado
naive_bayes_car_evaluation.fit(X_car_attributes_training, y_car_class_training)

## Teste
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

previsions_car_evaluation = naive_bayes_car_evaluation.predict(X_car_attributes_test)

accuracy_score(y_car_class_test, previsions_car_evaluation)
# Precisão 0.61


#  Matrix de confusão
from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(naive_bayes_car_evaluation)
cm.fit(X_car_attributes_training, y_car_class_training)
cm.score(X_car_attributes_test, y_car_class_test) # Ver gráfico
# Precisão 0.61


########### 02 - Decision Tree ###########
from sklearn.tree import DecisionTreeClassifier

tree_car_evaluation = DecisionTreeClassifier(criterion='entropy')

tree_car_evaluation.fit(X_car_attributes_training, y_car_class_training)

tree_car_evaluation.feature_importances_ # Mostra quais atributos tem mais importancia (qual elege como nó)
tree_car_evaluation.classes_

'''
from sklearn import tree
predictions = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
figure, axies = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
tree.plot_tree(tree_car_evaluation, feature_names=predictions, class_names=tree_car_evaluation.classes_, filled=True)
'''

predictions = tree_car_evaluation.predict(X_car_attributes_test)

accuracy_score(y_car_class_test, predictions)
# Precisão 0.98

#  Matrix de confusão
cm = ConfusionMatrix(tree_car_evaluation)
cm.fit(X_car_attributes_training, y_car_class_training)
cm.score(X_car_attributes_test, y_car_class_test)
# Precisão 0.98


########### 03 - Random Forest ###########
from sklearn.ensemble import RandomForestClassifier

random_forest_car_evaluation = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)

random_forest_car_evaluation.fit(X_car_attributes_training, y_car_class_training)

predictions = random_forest_car_evaluation.predict(X_car_attributes_test)

accuracy_score(y_car_class_test, predictions)
# Precisão 0.97

print(classification_report(y_car_class_test, predictions))

#  Matrix de confusão
cm = ConfusionMatrix(random_forest_car_evaluation)
cm.fit(X_car_attributes_training, y_car_class_training)
cm.score(X_car_attributes_test, y_car_class_test)
# Precisão 0.97


########### 04 - KNN ###########
from sklearn.neighbors import KNeighborsClassifier

knn_car_evaluation = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

knn_car_evaluation.fit(X_car_attributes_training, y_car_class_training)

predictions = knn_car_evaluation.predict(X_car_attributes_test)

accuracy_score(y_car_class_test, predictions)
# Precisão 0.92

print(classification_report(y_car_class_test, predictions))

#  Matrix de confusão
cm = ConfusionMatrix(knn_car_evaluation)
cm.fit(X_car_attributes_training, y_car_class_training)
cm.score(X_car_attributes_test, y_car_class_test)
# Precisão 0.92


########### 05 - SVM ###########
from sklearn.svm import SVC

svm_car_evaluation = SVC(C=5, kernel='rbf') # Se atentar ao parametro C

svm_car_evaluation.fit(X_car_attributes_training, y_car_class_training)

predictions = svm_car_evaluation.predict(X_car_attributes_test)

accuracy_score(y_car_class_test, predictions)
# Precisão 0.97

print(classification_report(y_car_class_test, predictions))

#  Matrix de confusão
cm = ConfusionMatrix(svm_car_evaluation)
cm.fit(X_car_attributes_training, y_car_class_training)
cm.score(X_car_attributes_test, y_car_class_test)
# Precisão 0.97


########### 06 - Neural Network ###########
from sklearn.neural_network import MLPClassifier

neural_network_car_evaluation = MLPClassifier(max_iter=1000, verbose=True)
neural_network_car_evaluation.fit(X_car_attributes_training, y_car_class_training)

predictions = neural_network_car_evaluation.predict(X_car_attributes_test)

accuracy_score(y_car_class_test, predictions)
# Precisão 0.98

print(classification_report(y_car_class_test, predictions))

#  Matrix de confusão
cm = ConfusionMatrix(neural_network_car_evaluation)
cm.fit(X_car_attributes_training, y_car_class_training)
cm.score(X_car_attributes_test, y_car_class_test)
# Precisão 0.98
