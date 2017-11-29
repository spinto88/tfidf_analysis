# -*- coding: utf-8 -*-

def tfidf_matrix(newspaper, init_date, final_date, section):

    import sqlite3
    from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
    import cPickle as pk
    import datetime

    tfidf = pk.load(open('idf.pk', 'r'))

    order = u'select distinct id, date, title, body from {}'.format(newspaper)
    if section != None:
        order += u' where section like "%{}%"'.format(section)
        order +=  u' and date >= "{}" and date < "{}" and title IS NOT NULL;'.format(init_date, final_date)
    else:
        order +=  u' where date >= "{}" and date < "{}" and title IS NOT NULL;'.format(init_date, final_date)

    conn = sqlite3.connect('../data.db')
    c = conn.cursor()
    c.execute(order)

    ids_relation = []
    content = []

    for row in c:
        try:
            content.append(row[2] + row[3])
        except:
            content.append(row[2])

        ids_relation.append({'db_id': row[0], \
            'date': datetime.datetime.strptime(row[1], "%Y-%m-%d").date()})

    conn.close()

    features = tfidf.vocabulary_.items()
    xtfidf = tfidf.transform(content)

    return xtfidf, features, ids_relation, content

def topics_estimation(xtfidf, features, delta = 0.20):

    import numpy as np

    N = xtfidf.shape[0]
    M = xtfidf.shape[1]
    nnz = list(xtfidf.getnnz(0))
    binc = np.bincount(nnz)

    density = [i * binc[i] for i in range(len(binc))]
    distribution = [np.sum(density[:i+1]) for i in range(len(density))]
    distribution = np.array(distribution, dtype = np.float)/np.max(distribution)

    inferior = min(range(len(distribution)), \
               key = lambda x: np.abs(distribution[x]-delta))

    superior = min(range(len(distribution)), \
               key = lambda x: np.abs(distribution[x]-1.00+delta))

    mp = M
    nnzp = np.sum(nnz)

    bcnnz = np.bincount(nnz)

    mp -= np.sum(binc[:inferior])
    nnzp -= np.sum([j * binc[j] for j in range(inferior)])
    mp -= np.sum(binc[superior:])
    nnzp -= np.sum([j * binc[j] for j in range(superior, len(distribution))])

    ntopics = int(N*mp/nnzp)

    features_filtered = filter(lambda x: nnz[x[1]] > inferior and \
                           nnz[x[1]] < superior, features)

    return ntopics, features_filtered, inferior, superior, density 

def nmf_decomposition(xtfidf, ntopics, random_seed = 123457):

    from sklearn.decomposition import NMF

    nmf = NMF(ntopics, random_state = random_seed)

    xnmf = nmf.fit_transform(xtfidf)

    return xnmf, nmf.components_

def principal_features(features, components, nprincipal = 10):

    pf = []
    for comp in components:

        pf_per_comp = sorted(features, reverse = True, key = lambda x: comp[x[1]])[:nprincipal]

        pf.append([x[0] for x in pf_per_comp])

    return pf

def save_features(foldername, features, components, nprincipal = 10, offset = 0):

    import codecs
    import os
    import csv 
    import cPickle as pk

    """
    Where to save the results
    """
    try:
        os.mkdir(foldername)
    except:
        pass
        
    pf = principal_features(features, components, nprincipal)

    for j in range(len(pf)):
        fp = codecs.open(foldername + '/features{}.txt'.format(j + offset), 'a', 'utf8')
        for k in pf[j]:
            fp.write(u'{}, '.format(k))
        fp.write('\n')
        fp.close()

    for j in range(len(pf)):
        fp = codecs.open(foldername + '/features.txt'.format(j + offset), 'a', 'utf8')
        for k in pf[j]:
            fp.write(u'{}, '.format(k))
        fp.write('\n')
        fp.close()

    """ Vector representation """
    for j in range(len(pf)):
        pk.dump(components[j], file(foldername + '/topic{}_vect.pk'.format(j + offset),'w'))

def save_temporal_profile(foldername, xnmf, ids_relation, content, offset = 0):

    import codecs
    import os
    import csv 

    """
    Where to save the results
    """
    try:
        os.mkdir(foldername)
    except:
        pass

    from sklearn.preprocessing import Normalizer
    import datetime
    import numpy as np
    from nltk.tokenize import word_tokenize
    
    norm1 = Normalizer('l1')

    xnmf = norm1.fit_transform(xnmf)

    ntopics = xnmf.shape[1]
    dates = sorted(list(set([idsr['date'] for idsr in ids_relation])))
    init_date = min(dates)
    final_date = max(dates)

    dates = []
    while init_date <= final_date:
        dates.append(init_date)
        init_date += datetime.timedelta(1)

    x_temp = np.zeros([ntopics, len(dates)], dtype = np.float)

    content_length = []
    for i in range(xnmf.shape[0]):
        content_length.append(len(word_tokenize(content[i])))

    # Calculo el peso de un topico como la cantidad de palabras de una nota
    # mas la proyeccion de esa nota al topico
    for j in range(ntopics):
      for i in range(xnmf.shape[0]):
        date_id = dates.index(ids_relation[i]['date'])
        x_temp[j][date_id] += content_length[i] * xnmf[i][j]

    for i in range(x_temp.shape[0]):
        with open(foldername + '/topic{}_temp.csv'.format(i + offset), 'w') as csvfile:
            csvfile.write('date,topic_weight\n')
            for j in range(len(dates)):
                csvfile.write('{},{}\n'.format(dates[j], x_temp[i][j]))
            csvfile.close()

    notes_topics = [np.argmax(x) for x in xnmf]
    for i in range(x_temp.shape[0]):
        with open(foldername + '/topic{}_idnotes.csv'.format(i + offset), 'w')\
                    as csvfile:
            csvfile.write('Database_ids_notes\n')
            for j in range(len(notes_topics)):
                if notes_topics[j] == i:
                    csvfile.write('{}\n'.format(ids_relation[j]['db_id']))
            csvfile.close()


def related_topics(foldername, topic, n_sigmas = 5):

    import cPickle as pk
    import numpy as np

    vector_topics = []
    """ Vector representation """
    for j in range(1000):
      try:
        vector_topics.append(pk.load(file(foldername + \
                             '/topic{}_vect.pk'.format(j),'r')))
      except:
        pass

    components_array = np.zeros([len(vector_topics), len(vector_topics[0])])
    for j in range(len(vector_topics)):
        components_array[j] = vector_topics[j]

    from sklearn.preprocessing import Normalizer
    norm2 = Normalizer('l2')
    comp_array = norm2.fit_transform(components_array)

    n_top = len(vector_topics)
    corrs = [comp_array[node1].dot(comp_array[node2]) \
             for node1 in range(n_top) for node2 in range(node1+1, n_top)]

    threshold = np.mean(corrs) + n_sigmas * np.std(corrs)

    related_topics = [i for i in range(n_top) \
                      if comp_array[topic].dot(comp_array[i]) > threshold]

    return related_topics    
