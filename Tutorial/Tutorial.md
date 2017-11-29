# Tutorial Topic Detection

## Instalacion de scikit-learn

Scikit-learn 
es un modulo de python para aplicaciones de algoritmos de machine learning.
La versión 0.19.1 es estable.
(__http://scikit-learn.org__) 

Instalación:
```
pip install scikit-learn==0.19.1
```

## Carga de textos

Los textos pueden ser cargados como una lista en python.
Por ejemplo, si los textos están guardado en archivos ***text(i).txt*** 
y hay ***Ndocumentos***:

```
import codecs 

# Lista de documentos
content = [codecs.open('text{}.txt'.format(i), "r", encoding = "utf-8").read() \
           for i in range(Ndocumentos)]
```

## Descripción matricial de los textos

#### Valorización IDF (Inverse Document Frequency)

Importo la clase TfidfVectorizer:
```
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
```
Creo el objeto con algunos parámetros relevantes:
```
tfidf = TFIDF(min_df = 2, max_df = 0.95, stop_words = stopwords, ngram_range = (1,3), norm = 'l2')
```
Los parámetros corresponden a:

- ***min_df (max_df)***: se descartan aquellos términos que aparecen en menos (más) 
  de **n** documentos, siendo **n** es un número entero.
  Si el parámetro es un número racional entre 0 y 1, denota la fracción del 
  contenido (por ejemplo, ***max_df*** = 0.95 indica que se descartan aquellos términos
  que aparezcan en más del 95% de los documentos).
- ***stop_words***: lista de palabras de comunes que se eliminan directamente del análisis.
- ***ngram_range***: indica si se consideran uni-gramas, bi-gramas, etc. En el ejemplo
  indica que se consideran desde uni-gramas hasta tri-gramas.
- ***norm***: norma de los vectores textos. Al aplicar el objeto sobre los textos
  devuelve el vector asociado ya normalizado a norma 2.

Para entrenar el objeto al contenido se utiliza:
```
tfidf.fit(content)
```

La valorización TFIDF se puede entrenar y ser utilizada posteriormente.
Se puede guardar el objeto entrenado con el módulo cPickle:
```
import cPickle as pk
pk.dump(tfidf, open('tfidf.pk','w'))
```
Para levantar el objeto posteriormente se utiliza:
```
tfidf = pk.load(file('tfidf.pk','r'))
```

#### Descripción TF-IDF (Term frequency - Inverse document frequency)

Para dar la descripción de un conjunto de documentos en el espacio
de términos, es decir crear la matriz documentos por términos 
se aplica:
```
xtfidf = tfidf.transform(content)
```
donde ***xtfidf*** es una matriz sparse.

Para consultar qué término corresponde a cada columna, se obtiene un 
diccionario de python con el comando:

```
features = tfidf.vocabulary_
```

El vocabulario queda definido al momento de entrenar el modelo 
(sin necesidad de obtener primero la matriz ***xtfidf***)

Es más cómodo para trabajar transformar el diccionario en una lista de pares
palabra-número de columna, haciendo:

```
features = features.items()
```

***features*** es una lista de pares, donde el primer elemento de cada par
es la palabra y el segundo el número de columna asociado.

## Análisis matricial

Con la descripción matricial de los textos se pueden aplicar cualquiera de la 
técnicas de descomposición matricial como PCA, NMF, SVD, etc.

#### Descomposición NMF

La descomposición en matrices no negativas (NMF) se puede realizar especificando
la cantidad de tópicos en los cuales hacer la descomposición. Realizando la 
misma se obtiene un nueva matriz de documentos por tópicos (***xnmf***):

```
from sklearn.decomposition import NMF

nmf = NMF(ntopics, random_state = seed)

xnmf = nmf.fit_transform(xtfidf)
```

La etiqueta asociada a cada documente se obtiene observando 
cuál es el componente más grande en el vector de documentos-tópicos:

```
# Tópico del primer documento
import numpy as np

print np.argmax(xnmf[0])
``` 

Los direcciones de los tópicos se pueden obtener de los componentes:

```
components = nmf.components_
```

Para obtener los términos ordenados de mayor a menor importancia se pueden aplicar 
los siguientes comandos:

```
# Creo una lista de términos
terms = [f[0] for f in features]

# Ordeno la lista de términos según los valores que estos adoptan, por ejemplo, en la primer dirección
principal_features_0 = \
      sorted(terms, reverse = True, key = lambda x: components[0][terms.index(x)])

# Imprimo los 5 términos principales:
for i in range(5):
    print principal_features_0[i],
```


## Código ejemplo

```python
# -*- coding: utf-8 -*-

import codecs
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.decomposition import NMF

# Ejemplo con 10 documentos en archivos de la forma text(i).txt
Ndocumentos = 10

content = [codecs.open('text{}.txt'.format(i), "r", encoding = "utf-8").read() \
           for i in range(Ndocumentos)]

# Creo el objeto tfidf
tfidf = TFIDF(min_df = 2, max_df = 0.95, stop_words = stopwords, \
              ngram_range = (1,3), norm = 'l2')

# Entreno con el contenido
tfidf.fit(content)

# Vocabulario
features = tfidf.vocabulary_.items()
terms = [f[0] for f in features]

# Creo la matriz documentos por términos 
xtfidf = tfidf.transform(content)

# Descomposición NMF en 2 tópicos
ntopics = 2
nmf = NMF(n = ntopics, random_state = 123457)

xnmf = nmf.fit_transform(xtfidf)

# Componentes y principales términos 
components = nmf.components_
for i in range(ntopics):

    principal_features = sorted(terms, reverse = True, \
                 key = lambda x: components[i][terms.index(x)])

    print 'Términos principales de la componente {}:'.format(i)
    for pf in principal_features[:5]:
        print pf,

    print '\n'

# Etiqueta de cada documento (máximo componente)
for i in range(Ndocuments):
    print 'Documento {} - Tópico asociado {}'.format(i, np.argmax(xnmf[i]))
```

    


