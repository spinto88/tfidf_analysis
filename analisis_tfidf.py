# -*- coding: utf-8 -*-

import auxiliar_functions as aux_func

# Direccion de la base de datos
database_path = 'data.db'

# Diario y section para analizar
newspaper = 'wshtimes'
section = "Politics"

# Rango de fechas donde consultar
init_date = '2017-01-02'
final_date = '2017-07-31'

# Carpeta donde guardar la informacion
foldername = '{}_{}'.format(newspaper, section)

# Analisis y devolucion de datos
xtfidf, features, ids_relation, content = \
                  aux_func.tfidf_matrix(database_path, newspaper, \
                  init_date, final_date, section) 

ntopics, features_filtered, inferior, superior, density = \
                  aux_func.topics_estimation(xtfidf, \
                  features, delta = 0.10)

print '# Topics estimated: {}'.format(ntopics)

xnmf, components = aux_func.nmf_decomposition(xtfidf, ntopics)

aux_func.save_features(foldername, features, components, nprincipal = 20)
aux_func.save_temporal_profile(foldername, xnmf, ids_relation, content)
