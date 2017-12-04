# -*- coding: utf-8 -*-

"""
Ejecutar en la terminal: python ejemplo.py
"""

import codecs
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.decomposition import NMF

# Ejemplo con 5 documentos en archivos de la forma text(i).txt
Ndocuments = 5

content = [codecs.open('texts/text{}.txt'.format(i), "r", encoding = "utf-8").read() \
           for i in range(Ndocuments)]

# Carga de stopwords en archivo externo
stopwords = codecs.open('stopwords_spanish.txt').read().split('\r\n')

tfidf = TFIDF(min_df = 2, max_df = 0.95, stop_words = stopwords, \
              ngram_range = (1,1), norm = 'l2')

# Entreno con el contenido
tfidf.fit(content)

# Vocabulario
features = tfidf.vocabulary_.items()
terms = [f[0] for f in features]

# Creo la matriz documentos por términos
xtfidf = tfidf.transform(content)

# Descomposición NMF en 2 tópicos
ntopics = 2
nmf = NMF(ntopics, random_state = 123457)

xnmf = nmf.fit_transform(xtfidf)

# Componentes y principales términos
components = nmf.components_
for i in range(ntopics):

    principal_features = sorted(terms, reverse = True, \
                 key = lambda x: components[i][terms.index(x)])

    print 'Términos principales de la componente {}:'.format(i)
    for pf in principal_features[:5]:
        print u'{}, '.format(pf),

    print '\n'

# Etiqueta de cada documento (máximo componente)
for i in range(Ndocuments):
    print 'Documento {} - Tópico asociado {}'.format(i, np.argmax(xnmf[i]))

