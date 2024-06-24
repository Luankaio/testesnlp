import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag, pos_tag_sents

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer  #criar tf-idf
import re
import string
import wordcloud
import pandas as pd
from nltk.draw.dispersion import dispersion_plot
import matplotlib.pyplot as plt



nltk.download("stopwords")
nltk.download("punkt")
nltk.download("tagsets")
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
nltk.download('maxent_ne_chunker')

spam = pd.read_csv('output.csv')
spam.head()

spam.dropna(axis=0, inplace=True)


def pre_processamento(texto):
    # Seleciona apenas letras e coloca todas em minúsculo
    letras_min = re.findall(r'\b[A-zÀ-úü]+\b', texto.lower())

    # Remove stopwords
    stop = set(stopwords.words('english'))
    sem_stopwords = [w for w in letras_min if w not in stop]

    # Juntando os tokens novamente em formato de texto
    texto_limpo = " ".join(sem_stopwords)

    return texto_limpo


def cleanResume(resumeText):
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) # remove non-ascii characters
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    resumeText = re.sub(r'[0-9]+', '', resumeText)  #remy_testeove numbers
    return resumeText.lower()


import nltk
nltk.download('stopwords')
import string
from nltk.corpus import stopwords
from nltk import word_tokenize

len(stopwords.words('english'))

#spam["text"] = spam["text"].apply(lambda x: cleanResume(x))
#len(spam["text"][1])

spam['text'] = spam['text'].apply(pre_processamento)

#spam['text'] = spam['text'].apply(lambda x: " ".join(x) if isinstance(x, list) else x)

#from gensim.models import Word2Vec
#import numpy as np
#model = Word2Vec(sentences=spam['text'], vector_size=100, window=5, min_count=1, workers=4)

#def get_average_word2vec(tokens_list, model, vector_size):
#    # Obter os vetores de cada palavra e calcular a média
#    vectors = [model.wv[word] for word in tokens_list if word in model.wv]
#    if len(vectors) > 0:
#        return np.mean(vectors, axis=0)
#@    else:
 #       return np.zeros(vector_size)
 
 #vector_size = model.vector_size
#spam['text'] = spam['text'].apply(lambda tokens: get_average_word2vec(tokens, model, vector_size))

#previsores = np.vstack(spam['text'].values)
#classe = spam['class'].values

#token = word_tokenize(spam['text'])#tokenizar
#spam['text'] = spam['text'].apply(word_tokenize)


#tokenlemm = [lemmatizer.lemmatize(palavra) for palavra in token]

previ = spam['text']
classe = spam['job']
misto = spam['text']
previ


###precisamos vetorizar pois o modelo de machine learn so entende numeros
#vetorizamos usando o modelo tf-idf que correlaciona com peso
#pesquisar o funcionamento desses comandos
#vetorizador = TfidfVectorizer()
#previsores = vetorizador.fit_transform(previ)  #fit vai criar o modelo e o transforme vai fazer a transformação


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
previsores = vectorizer.fit_transform(previ)


#vetorizador2 = TfidfVectorizer()
#previsores2 = vetorizador2.fit_transform(spam['skills'])

#from transformers import BertTokenizer, BertModel
#import torch
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertModel.from_pretrained('bert-base-uncased')

#def get_bert_embeddings(texts):
#    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
#    with torch.no_grad():
#        outputs = model(**inputs)
#    return outputs.last_hidden_state.mean(dim=1)  # Usar a média dos embeddings das palavras
#def get_bert_embeddings_batch(texts, batch_size=16):
#    all_embeddings = []
#    for i in range(0, len(texts), batch_size):
#        batch_texts = texts[i:i+batch_size]
#        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
#        with torch.no_grad():
#            outputs = model(**inputs)
#        batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Usar a média dos embeddings das palavras
#        all_embeddings.append(batch_embeddings)
#    return torch.cat(all_embeddings)


#embeddings = get_bert_embeddings_batch(previ.tolist(), batch_size=16)
#previsores = embeddings.numpy()

#skills = spam['skills']

#import scipy.sparse
#X = scipy.sparse.hstack((previsores, previsores2))

#revisores.shape
#print(vetorizador.get_feature_names_out()[10:100])

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores, classe, test_size=0.2)
#train_test_split faz a do modelo
#previsores são as frases para ele treinar
#classe é o resultado de cada mensagem e test_size é o tamanho separado para testes

#floresta = RandomForestClassifier(n_estimators=100)
#floresta.fit(X_treinamento, y_treinamento) #identifica de acordo com o x_treinamento qual o y_treinamento vai vir

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_treinamento, y_treinamento)

y_test_pred = logreg.predict(X_teste)
y_pred = logreg.predict(X_teste)

accuracy_score(y_teste, y_test_pred)


from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier()
gb_model.fit(X_treinamento, y_treinamento)

y_test_pred = gb_model.predict(X_teste)
y_pred=gb_model.predict(X_teste)

accuracy_score(y_teste, y_test_pred)

