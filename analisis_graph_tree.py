import cPickle as pk
from sklearn.preprocessing import Normalizer
import numpy as np
import matplotlib.pyplot as plt
import math

newspaper = 'lanacion'
ntopics = 64

norm2 = Normalizer('l2')

first_topic = pk.load(file("../{}/topic{}_vect.pk".format(newspaper, 0), 'r'))
topics = np.zeros([ntopics, first_topic.shape[0]])

for i in range(ntopics):
    topics[i] = pk.load(file("../{}/topic{}_vect.pk".format(newspaper, i), 'r'))

topics = norm2.fit_transform(topics)

sim = [[i, j, topics[i].dot(topics[j])]\
        for i in range(ntopics) for j in range(i+1, ntopics)]

sim = sorted(sim, reverse = True, key = lambda x: x[2])

import networkx as nx

graph = nx.empty_graph(ntopics)

for edge in sim:

  graph.add_edge(edge[0], edge[1], weight = edge[2])

  if len(nx.cycle_basis(graph)) != 0:
      graph.remove_edge(edge[0], edge[1])
  else:
     pass

edges = graph.edges()

fp = open("Edge_list.txt","w")
for edge in edges:
   fp.write("{} {} {}\n".format(edge[0], edge[1], graph.edge[edge[0]][edge[1]]['weight']))
fp.close()

import igraph

graph = igraph.Graph.Read_Ncol("Edge_list.txt", directed = False, weights = True)

vertex_size = []

for i in range(ntopics):

    fp = open("../{}/topic{}_idnotes.csv".format(newspaper, i), 'r')
    notes = fp.read()
    fp.close()
    
    vertex_size.append(len(notes.split('\n')) - 1)

vertex_size = 40 * (np.array(vertex_size, dtype = np.float) - np.min(vertex_size)) / \
		(np.max(vertex_size) - np.min(vertex_size)) + 10
               
"""
vertex_size = [10 + 15 * np.sum([es['weight'] for es in graph.es() \
                       if es.target == vs.index \
                       or es.source == vs.index]) for vs in graph.vs]
"""

igraph.plot(graph, vertex_size = vertex_size, vertex_label = [vs['name'] for vs in graph.vs], vertex_label_size = 8, vertex_color = 'cyan', edge_width = [es['weight'] * 10 for es in graph.es()], target = "{}_layout{}.pdf".format(newspaper, ntopics))

"""
igraph.plot(graph, vertex_size = vertex_size, vertex_label = [vs['name'] for vs in graph.vs], vertex_label_size = 8, vertex_color = 'cyan', edge_width = [es['weight'] * 10 for es in graph.es()])
"""
