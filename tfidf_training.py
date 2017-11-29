# -*- coding: utf-8 -*-
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
import cPickle as pk
import codecs

init_date = "2016-01-01"
final_date = "2017-08-01"

database_path = 'data.db'

newspapers = ['nytimes', shtimes']

conn = sqlite3.connect(database_path)
c = conn.cursor()

c.execute(u'select title, body from nytimes where title IS NOT NULL and body IS NOT NULL and date >= "{}" and date <= "{}";'.format(init_date, final_date))
content = [row[0] + row[1] for row in c]
c.execute(u'select title, body from wshtimes where title IS NOT NULL and body IS NOT NULL and date >= "{}" and date <= "{}";'.format(init_date, final_date))
content = [row[0] + row[1] for row in c]

conn.close()

#Palabras comunes
import codecs
fp = codecs.open("stopwords.txt", "r", encoding = "utf-8")
data = fp.read()
fp.close()
aux = data.split('\n')
words = [a.lower() for a in aux]

"""
# Entrenamiento de la valorizacion tfidf
"""
tfidf = Tfidf(min_df = 2, max_df = 0.95, \
              stop_words = words, \
              ngram_range = (1,1))

tfidf.fit(content)

pk.dump(tfidf, open('idf.pk','w'))
