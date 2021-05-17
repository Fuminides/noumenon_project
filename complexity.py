# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 13:18:45 2019
@author: Javier Fumanal Idocin.

This modeule contains the functions that compute the semantic value for an actor,
and for all the actors in a directed network.

I know the documentation is a little bit scarce, but I hope to fix sometime. 
I could really use a hand if yo are reading this.

Please, cite any of my works with semantic value if yo use this code.
"""
import networkx as nx
import numpy as np
import pandas as pd


def semantics_node_directed(df_affinity, intrinsic_value, node):
    '''
    Calculates the semantic value of a node in a directed network.
    '''
    conexiones = df_affinity.loc[:, node] > 0
    conexiones[node] = False
    
    grados = intrinsic_value[conexiones]
    grados = grados.reshape((len(grados),))
    nombres_conexion = conexiones.index[conexiones]
    
    semantica_transmitida_bruta = np.multiply(df_affinity.loc[:, node][conexiones], grados)
    semantica_correlada = np.zeros(semantica_transmitida_bruta.shape)
    for ix, otro in enumerate(nombres_conexion):
        if conexiones[ix]:
            entradas_otro = df_affinity.loc[otro, :] > 0
            entradas_otro[otro] = False
            
            conexiones_conjuntas = np.logical_and(entradas_otro, conexiones)
            semantica_correlada[ix] += np.sum(df_affinity.loc[conexiones_conjuntas, node] * df_affinity.loc[otro, conexiones_conjuntas] * semantica_transmitida_bruta[conexiones_conjuntas])
    
    transmitidos = semantica_transmitida_bruta - semantica_correlada
    
    return np.sum(transmitidos[transmitidos > 0])# + intrinsic_value[df_affinity.columns == node]

def count_words(file, encoding='cp1252'):
    '''
    Returns the word histogram from a text file.
    '''
    from nltk.stem import PorterStemmer
    import re

    ps = PorterStemmer() 
    
    # Open the file in read mode 
    text = open(file, "r", encoding=encoding) 
      
    # Create an empty dictionary 
    d = dict() 
      
    # Loop through each line of the file 
    for line in text: 
        # Remove the leading spaces and newline character 
        line = line.strip() 
      
        # Convert the characters in line to  
        # lowercase to avoid case mismatch 
        line = line.lower() 
      
        # Split the line into words 
        words = line.split(" ") 
      
        # Iterate over each word in line 
        for word in words: 
            word = re.sub(r'\W+', '', word)
            if len(ps.stem(word)) > 1:
                # Check if the word is already in dictionary 
                if ps.stem(word) in d: 
                    # Increment count of word by 1 
                    d[ps.stem(word)] = d[ps.stem(word)] + 1
                else: 
                    # Add the word to dictionary with count 1 
                    d[ps.stem(word)] = 1
        
    return d

def dict_to_csv(dic, output_path):
    '''
    Writes a dictionary as a csv.

    Parameters
    ----------
    dic : TYPE
        DESCRIPTION.
    output_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    import csv
    with open(output_path,'w') as f:
        w = csv.writer(f)
        w.writerows(dic.items())

def semantics_network(df_affinity, intrinsic_value, external_too=False):
     '''
     Returns the semantic value for each node.
     '''
     entities = list(df_affinity)
     result = {}
     if external_too:
         external = {}
     for ix, entity in enumerate(entities):
         if external_too:
            external[entity] = semantics_node_directed(df_affinity, intrinsic_value, entity)
            result[entity] = external[entity] + intrinsic_value[ix]
         else:
            result[entity] = semantics_node_directed(df_affinity, intrinsic_value, entity) + intrinsic_value[ix]


     if external_too:
        return result, external
     else:
        return result

def table_for_paper(mythh): #THIS REPLICATES EXACTLY THE EXPERIMENT FROM THE PAPER.
    import speech_to_adjacency as sa
    import complexity as cm
    import affinity as af

    top = 15
    st_fields = ['$S$', '$E$', 'Freq. ($I$)', 'Degree', 'Betweenness', 'Closeness', 'Eigencentrality']
    res = pd.DataFrame(np.zeros((top, len(st_fields))), columns = st_fields)

    if mythh == sa.edda:
        mm = sa.canon_edda()
        entities = ['Odin', 'Name', 'Thor', 'Loke', 'Hvergelmer', 'Son', 'Dwarf', 'Sigurd', 'Frey', 'Freyja' ] #For commodity I hardcoded them here. They are just the most frequent names.
    elif mythh == sa.greek:
        mm = sa.canon_greek()
        entities = ['Heracles', 'Theseus', 'Apollo', 'Psyche', 'Jason', 'Eurystheus', 'Zeus', 'Perseus', 'Man', 'Pelias']
    elif mythh == sa.celtic:
        mm = sa.canon_celt()
        entities = ['Ireland', 'Lugh', 'Balor', 'Turann', 'Conary', 'King', 'Son', 'Gobhaun', 'Earth', 'Dagda']
    elif mythh == 'myth':
        mm = sa.canon_myth_all()
        entities = ['Odin', 'Thor', 'Name', 'Son', 'Heracles', 'Loke', 'Hvergelmer', 'Lugh', 'King', 'Ireland']

    if mythh != 'myth':
        #n_words, n_entities, most_important, hist_entities = sa.basic_book_analysis(sa.load_known_libro(mythh))
        text = sa.load_known_libro(mythh)
        text_words = text.split()
        hist_entities = {}
        for word in text_words:
            try:
                hist_entities[word] += 1
            except KeyError:
                hist_entities[word] = 1


    else:
        hist_entities = {}

        text = sa.load_known_libro(sa.edda)
        text_words = text.split()

        for word in text_words:
            try:
                hist_entities[word] += 1
            except KeyError:
                hist_entities[word] = 1

        text = sa.load_known_libro(sa.celtic)
        text_words = text.split()

        for word in text_words:
            try:
                hist_entities[word] += 1
            except KeyError:
                hist_entities[word] = 1

        text = sa.load_known_libro(sa.greek)
        text_words = text.split()

        for word in text_words:
            try:
                hist_entities[word] += 1
            except KeyError:
                hist_entities[word] = 1

    adjacency_df = sa.edges_df_to_adjacency_df(mm, symmetry=True)
    degrees = np.sum(adjacency_df, axis=1)
    intrinsic_value = np.zeros(adjacency_df.shape[0])
    for ix, x in enumerate(adjacency_df.index):
        try:
            intrinsic_value[ix] = hist_entities[x]
        except KeyError:
            intrinsic_value[ix] = 1

    for ix, x in enumerate(intrinsic_value):
        if x == 0:
            intrinsic_value[ix] = 1

    df_affinity = adjacency_df.copy()
    df_affinity[:] = af.connexion2affinity_important_friend(df_affinity.values)
    bet = nx.betweenness_centrality(nx.from_pandas_adjacency(df_affinity>0))
    clos = nx.closeness_centrality(nx.from_pandas_adjacency(df_affinity>0))
    eig = nx.eigenvector_centrality(nx.from_pandas_adjacency(df_affinity>0))
    
    semantics, extrinsic = cm.semantics_network(df_affinity, intrinsic_value, external_too=True)
    sa.semantics_dict_csv(semantics, mythh + '_full.csv')
    entities = np.argsort(intrinsic_value)[::-1]
    
    study_group = [df_affinity.index[x] for x in entities[0:top]]
    res.index = study_group
    res.loc[:, res.columns[0]] = [semantics[k] for k in study_group]
    res.loc[:, res.columns[1]] = [extrinsic[k] for k in study_group]
    res.loc[:, res.columns[2]] = [int(hist_entities[k]) for k in study_group]
    res.loc[:, res.columns[3]] = [int(degrees[entities[ix]]) for ix, k in enumerate(study_group)]
    res.loc[:, res.columns[4]] = [float(bet[k]) for ix, k in enumerate(study_group)]
    res.loc[:, res.columns[5]] = [float(clos[k]) for ix, k in enumerate(study_group)]
    res.loc[:, res.columns[6]] = [float(eig[k]) for ix, k in enumerate(study_group)]
    res = res.sort_values(by=st_fields[0], ascending=False)
    res.to_latex('tab_' + mythh + '.tex', escape=False, float_format="%.2f", column_format='l' + 'c'*len(st_fields))
    
    return res

if __name__ == '__main__':
    import speech_to_adjacency as sa

    table_for_paper('myth')
    table_for_paper(sa.edda)
    table_for_paper(sa.celtic)
    table_for_paper(sa.greek)
    