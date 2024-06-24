import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import yake
import numpy as np




spam = pd.read_csv('output.csv')

spam.dropna(axis=0, inplace=True)

previ = spam['text']
classe = spam['class']

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
previsores = vectorizer.fit_transform(previ)
vectorizer2 = TfidfVectorizer()
tfidf_matrix = vectorizer2.fit_transform(previ)

print(1)
previsores_count_dense = previsores.toarray()
previsores_tfidf_dense = tfidf_matrix.toarray()

C = previsores_count_dense * previsores_tfidf_dense

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

print(1)
        
X_train, X_test, y_train, y_test = train_test_split(previsores, classe, test_size=0.3, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(tfidf_matrix, classe, test_size=0.3, random_state=42)
X_train3, X_test3, y_train3, y_test3 = train_test_split(C, classe, test_size=0.3, random_state=42)


floresta = RandomForestClassifier(n_estimators=100) #faz 100 testes
floresta.fit(X_train, y_train) #faz o treinamento do modelo
previsoes = floresta.predict(X_test)
print(1)
floresta2 = RandomForestClassifier(n_estimators=100) #faz 100 testes
floresta2.fit(X_train2, y_train2) #faz o treinamento do modelo
previsoes2 = floresta2.predict(X_test2)
print(1)

floresta3 = RandomForestClassifier(n_estimators=100) #faz 100 testes
floresta3.fit(X_train3, y_train3) #faz o treinamento do modelo
previsoes3 = floresta3.predict(X_test3)
print(accuracy_score(y_test, previsoes))
print(accuracy_score(y_test2, previsoes2))
print(accuracy_score(y_test3, previsoes3))

print(12)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf_model = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=2, scoring='accuracy', n_jobs=-1)
print(1)

grid_search.fit(X_train, y_train)
print(1)

print("Melhores hiperparâmetros encontrados:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
print("Acurácia no conjunto de teste:", accuracy)