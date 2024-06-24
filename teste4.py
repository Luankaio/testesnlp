import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import yake
import nltk
from nltk.corpus import stopwords
import re   
nltk.download("stopwords")


spam = pd.read_csv('output.csv')

spam.dropna(axis=0, inplace=True)

previ = spam['text']
classe = spam['class']

def pre_processamento(texto):
    # Seleciona apenas letras e coloca todas em minúsculo
    letras_min = re.findall(r'\b[A-zÀ-úü]+\b', texto.lower())

    # Remove stopwords
    stop = set(stopwords.words('english'))
    sem_stopwords = [w for w in letras_min if w not in stop]

    # Juntando os tokens novamente em formato de texto
    texto_limpo = " ".join(sem_stopwords)

    return texto_limpo


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
previsores = vectorizer.fit_transform(previ)
def calcular_yake(textos):
    kw_extractor = yake.KeywordExtractor(lan="en")
    yake_weights = []
    for texto in textos:
        keywords = kw_extractor.extract_keywords(texto)
        yake_weights.append({kw: score for kw, score in keywords})
    return yake_weights

def calcular_tfidf(textos):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(textos)
    words = vectorizer.get_feature_names_out()
    tfidf_weights = tfidf_matrix.toarray()
    return words, tfidf_weights

words, tfidf_weights = calcular_tfidf(previ)
yake_weights = calcular_yake(previ)
num_words = len(words)
words_index = {word: idx for idx, word in enumerate(words)}

import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



combined_weights = np.zeros((len(previ), num_words))
for i, texto in enumerate(previ):
    tfidf_row = tfidf_weights[i]
    yake_row = yake_weights[i]
    
    for word, idx in words_index.items():
        tfidf_score = tfidf_row[idx]
        yake_score = yake_row.get(word, 0)
        combined_weights[i, idx] = tfidf_score + (1 - yake_score)

        
X_train, X_test, y_train, y_test = train_test_split(combined_weights, classe, test_size=0.3, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(combined_weights, classe, test_size=0.3, random_state=42)



from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

y_test_pred = gb_model.predict(X_test)
y_pred=gb_model.predict(X_test)

accuracy_score(y_test, y_test_pred)

print(accuracy_score(y_test, y_pred))
