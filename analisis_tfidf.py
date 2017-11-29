# -*- coding: utf-8 -*-

import auxiliar_functions as aux_func

newspaper = 'wshtimes'
section = "Politics"

foldername = '{}_{}'.format(newspaper, section)

init_date = '2017-01-02'
final_date = '2017-07-31'

xtfidf, features, ids_relation, content = aux_func.tfidf_matrix(newspaper, \
                                            init_date, final_date, section) 

ntopics, features_filtered, \
             inferior, superior, density = aux_func.topics_estimation(xtfidf, \
                                 features, delta = 0.10)
print ntopics

xnmf, components = aux_func.nmf_decomposition(xtfidf, ntopics)

aux_func.save_features(foldername, features, components, nprincipal = 20)
aux_func.save_temporal_profile(foldername, xnmf, ids_relation, content)
