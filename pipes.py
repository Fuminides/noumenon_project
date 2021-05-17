import re

import networkx as nx
import numpy as np
import pandas as pd

from scipy import sparse

import speech_to_adjacency as sa
import complexity as cm
import affinity as af

# -*- coding: utf-8 -*-
'''
Created on I dont remember.

@author: Javier Fumanal Idocin

This module is used to compute the pipe algorithm (and therefore, the semantic affinity)
'''
def top_k(arr, k):
    '''
    Return the top k elements from a list.
    '''
    return arr.argsort()[-k:][::-1]

def full_process_csv(csv_edges, alpha, symmetric = True, csr = False):
    '''
    inputs:
        csv_edges -> a loaded csv file with the edges in the format 'Source' 'Target' 'Weight'
    
    returns:
        
        
    '''
    entidades = np.unique(csv_edges[['Source', 'Target']])
    num_entidades = len(entidades)
    
    diccionario = {}
    diccionario_reverso = {}

    identificador = 0
    for i in range(0, len(entidades)):
        diccionario[entidades[i]] = identificador
        diccionario_reverso[identificador] = entidades[i]
        identificador += 1

    if not csr:
        A = np.zeros([num_entidades, num_entidades], np.float64)
    else:
        A = sparse.lil_matrix((num_entidades, num_entidades), dtype=np.float64)

    names = list(diccionario.keys())
    
    for i in range(0, csv_edges.shape[0]):
        source = csv_edges.iloc[i][0]
        target = csv_edges.iloc[i][1]
        try:
            peso = csv_edges.iloc[i][2]
        except IndexError:
            peso = 1
        A[diccionario[source], diccionario[target]] = peso

        if symmetric:
            A[diccionario[target], diccionario[source]] = peso #Comment this to load eurovision

    
    linkeos = []
    if not csr:
        alpha = 0.1
        if (alpha != 1) and (alpha != 0):
          data = af.connexion2affinity_important_friend(A)*alpha + af.connexion2affinity(A, af.affinity_maquiavelo)*(1-alpha)*(af.connexion2affinity_important_friend(A)>0)
        elif alpha == 1:
          data = af.connexion2affinity_important_friend(A)*alpha + np.identity(A.shape[0])
        else:
          data = af.connexion2affinity(A)*(1-alpha) + np.identity(A.shape[0])
          
        p2 = data.copy()
        p2 = pd.DataFrame(p2)
        p2.columns = names
        p2.index = names
    else:
        af.connexion2affinity_important_friend(A, csr) + sparse.eye(A.shape[0])
        p2 = None

    masses2 = np.sum(A>0,axis=1).flatten()

    di = diccionario.copy()
    for ix, i in enumerate(di.keys()):
        if masses2.shape[0] == 1:
            di[i] = (masses2[0, ix])
        else:
            di[i] = (masses2[ix])
        
    
    return data, A, p2, di, diccionario, diccionario_reverso, linkeos

def study_case(df_affinity, semantics, source, target):
    '''
    Prints whether there is path or not between two nodes and their affinity.
    '''
    G2 = build_pipe_graph(df_affinity.T, semantics)
    print(source, semantics[source])
    print(target, semantics[target])
    print(target + ' -> ' + source, df_affinity.loc[target, source])
    print(source + ' -> ' + target, df_affinity.loc[source, target])
    try:
        print('edge ' + source + ' -> ' + target, G2.edges[source, target])
    except KeyError:
        print('Non-existing edge')

    try:
        paths_generator = nx.all_shortest_paths(G2, source, target, 10)
        try:
            print(next(paths_generator))
        except TypeError:
            print('list', list(paths_generator))
        except StopIteration:
            print('No path!')

    except nx.NetworkXNoPath:
        print('No path!')


def build_pipe_graph(df_affinity, semantics, affinity_mode=False):
    '''
      Builds a new graph in the networkx format. Each edge represents the pipe capacity
      for each pair of nodes:
      W(edge(x,y)) = semantic(x) * aff(x,y)
    '''
    G2 = nx.from_numpy_matrix(df_affinity.values, parallel_edges=False, 
                         create_using=nx.MultiDiGraph())
    label_mapping = {x:df_affinity.index[x] for x in range(df_affinity.shape[0])}
    G2 = nx.relabel_nodes(G2, label_mapping)

    max_weight = 1
    
    if not affinity_mode:
        for edge in G2.edges():
            intro = edge[0]
            outro = edge[1]
            
            if intro != outro:
                aux = G2[intro][outro][0]['weight'] * semantics[intro]
                G2[intro][outro][0]['weight'] = aux
                if aux > max_weight:
                    max_weight = aux
    
    for edge in G2.edges():
        intro = edge[0]
        outro = edge[1]
        
        if intro != outro:
            G2[intro][outro][0]['weight'] = max_weight - G2[intro][outro][0]['weight']#abs(G2[intro][outro][0]['weight'] - max_weigth)
                

    return G2


def llenar_camino(c, af_df, capacidades, visitados):
    '''
    Fills the path c using the capacities of each node.

    :param c: node list. (Path)
    :param af_df: affinity df
    :param capacidades: list of remaining capacities for each node in c.
    :param visitados: list of nodes already filled.
    '''
    len_camino = len(c) - 1
    camino_transportado = 0
    valor_transportar = capacidades[c[0]]
    # print('Valor a transportar: ', valor_transportar)
    # print('Camino: ', c)
    for i in range(len_camino):
        af_camino = af_df.loc[c[i], c[i + 1]]
        # print(capacidades[c[i+1]], af_camino*valor_transportar)
        capacidad = min(capacidades[c[i + 1]], af_camino * valor_transportar)
        capacidades[c[i + 1]] = max(0, capacidades[c[i + 1]] - af_camino * valor_transportar)
        camino_transportado += capacidad
        visitados.add(c[i + 1])

    capacidades[c[0]] -= camino_transportado
    # print('He llevado esta vez: ', camino_transportado)
    return visitados, camino_transportado


def pipe_comparison(af_df, liquid, source, target, max_len=20, affinity_mode=False, local_scale=False):
    '''
    Computes the affinity for the pair of nodes source and target. Discards
    possible paths that are bigger than max_len.

    :param af_df: Affinity network in a dataframe matrix format.
    :param liquid: Semantic values.
    :param source: Emiter
    :param target: Receptor
    :param max_len: discard paths bigger than this in the affinity computation. (Saves time)
    '''
    print(source, target)
    if source == target:
        return 1
    G2 = build_pipe_graph(af_df.T, liquid, affinity_mode)
    
    #whole_paths = []
    init_value = liquid[source]
    valor_transportar = liquid[source]
    capacidad = liquid[target]

    
    nodos_visitados = set()
    if affinity_mode:
        afinidades_camino = []
    resting_capacity = {(u, v): liquid[u]* af_df.loc[u, v] for u, v, a in G2.edges(data=True)}
    valor_transportar = liquid[source]
    
        
    for ix in range(500):     
        try:
             path = nx.shortest_path(G2, source, target, 'weight')
             valor_transportar = liquid[source]
             if len(path) > 0:
               for i in range(len(path)-1):
                     #af_camino = af_df.loc[path[i], path[i + 1]]
                     valor_transportar = min(liquid[path[i + 1]], valor_transportar, resting_capacity[path[i], path[i + 1]])
                     resting_capacity[path[i], path[i + 1]] = max(0, resting_capacity[path[i], path[i + 1]] - valor_transportar)
               
             
               #whole_paths.append(path)
               for i in range(len(path) - 1):
                   if resting_capacity[path[i], path[i + 1]] <= 0:
                       G2.remove_edge(path[i], path[i + 1])
                   
               nodos_visitados_iter, cap_origen_nueva = llenar_camino(path, af_df, liquid.copy(), nodos_visitados)
               print('Pendiente: ' + str(liquid[source]))
               print('Transportado esta vez: ' + str(valor_transportar))
               
               liquid[source] -= valor_transportar
               
               nodos_visitados = nodos_visitados.union(path)
               if affinity_mode:
                   for ix, x in enumerate(path):
                       if ix + 1 < len(path):
                           afinidades_camino.append(af_df.loc[path[ix], path[ix + 1]])
                           
               #print('Nodos visitados ' + str(nodos_visitados) + ' (' +str(len(nodos_visitados)) + ')')
               
               eficacia_threshold = 0.00001
               
               if not affinity_mode:
                   eficacia_path = valor_transportar / sum([liquid[x] for x in path])
               else:
                  eficacia_threshold = 0.0000
                  eficacia_path = np.mean([af_df.loc[path[a], path[a + 1]] for a in range(len(path) - 1)])
                  
               if liquid[source] <= max(liquid[source] - liquid[target], 0) or (eficacia_path < eficacia_threshold):
                   break
             else:
               break
        except nx.NetworkXNoPath:
             return 0


    #if len(whole_paths) == 0:
    #    return 0

    #whole_paths.sort(key=len)
    
    '''for c in whole_paths:
        
        if liquid[source] < 1:
            break
    try:
        nodos_visitados.remove(source)
    except KeyError:
        pass
    '''
    
    residuo = liquid[source]#HAY QUE CAMBIAR ESTO EN EL MODO AFINIDAD ES LA MEDIA DE LAS AFINIDADES DE LOS CAMINOS RECORRIDOS EN NODOS VISITADOS
    if affinity_mode:
        if not local_scale:
            afinidad_media = np.mean(afinidades_camino)
            return afinidad_media * (1 - residuo / init_value)
        else:
            afinidad_media = np.mean(afinidades_camino)
            max_affinity = max(afinidades_camino)
            
            return (1 - residuo / init_value) * afinidad_media / max_affinity
    else:
        nodos_visitados.remove(target)
        
        capacidad_usada = sum([liquid[x] for x in nodos_visitados])
        return (1 - residuo / init_value) * ( init_value - residuo ) / (capacidad_usada)
        

def compute_pipe_distances(red, df_affinity, semantics, entities=None, top=25, original_degree=None, verbose=False, affinity_mode=False, local_af=False):
    '''
    Computes the pipe affinity for each pair of nodes.

    :param red: networkx net containing affintiy edges.
    :param df_affinity: DataFrame of the affinity for each edge.
    :param semantics: Intrinsic value for each one.
    :param entities: entities to compute (top parameter by default)
    :param top: if we do not specify entities, we compute the top k ones according to semantics (25 by default)
    :param verbose: if true prints the affinity and path distance for each pair of nodes.
    '''

    if entities is None:
        if original_degree is None:
            entities = [a[0] for a in sorted(semantics.items(), key=lambda x: x[1])[::-1][0:top]]
        else:
            entities = [a[0] for a in sorted(original_degree.items(), key=lambda x: x[1])[::-1][0:top]]

    aumento = 0
    distances = pd.DataFrame(np.zeros((len(entities), len(entities))) )
    distances.columns = entities
    distances.index = entities

    for source in entities:
        for target in entities:
            if source == target:
                distances.loc[source, target] = 1
            else:
                distances.loc[source, target] = pipe_comparison(df_affinity, semantics.copy(), source, target, affinity_mode=affinity_mode, local_scale=local_af)
                print(distances.loc[source, target])
            if verbose:
                study_case(df_affinity, semantics, source, target)
                print('')

    report_df = pd.DataFrame(np.zeros((top, 3)), index=entities, columns=['$S$', 'Degree', 'Text Appearances'])
    for entit in entities:
        report_df.loc[entit] = {report_df.columns[0]: semantics[entit], report_df.columns[1]: original_degree[entit], report_df.columns[2]: 0}
        
    return distances, report_df

def pipe_preliminars(edge_list0, trh=10, preprocess=False, alpha = 0.1):
    '''
    Loads the specified edge_list.

    Returns networkx net, prunned edge_list, array affinity, array adjacency, data frame affinity, dict intrinsic value.
    '''
    import hashlib
    import affinity as af
    if isinstance(edge_list0, str):
        edge_list0 = sa.load_edges(edge_list0, True)

    if preprocess:
        edge_list = edge_list0.copy()
        edge_list = edge_list[edge_list['Source'] != edge_list['Target']]
        edge_list = edge_list[edge_list['Source'].apply(len) > 2]
        edge_list = edge_list[edge_list['Target'].apply(len) > 2]
        edge_list['Source'] = edge_list['Source'].apply(lambda a: re.sub(r'\W+', '', a))
        edge_list['Target'] = edge_list['Target'].apply(lambda a: re.sub(r'\W+', '', a))
        indices = [x for x in range(edge_list.shape[0]) if edge_list.iloc[x]['Weight'] > trh]
        edge_list = edge_list.iloc[indices, :]
    else:
        edge_list = edge_list0.copy()


    alpha = 0.9; beta = 0.0
    array_affinity, array_connection, df_affinity, diccionario_masas, dic, rdic, link = full_process_csv(edge_list, alpha,
                                                                                                            csr=False,
                                                                                                            symmetric=True)
    
    aux = af.connexion2affinity_important_friend(array_connection) * alpha
    aux = aux + (1 - alpha - beta) * af.connexion2affinity(array_connection, af.affinity_maquiavelo) * (af.connexion2affinity_best_common_friend(array_connection)>0)
    df_affinity = pd.DataFrame(aux, index=df_affinity.index, columns=df_affinity.columns) 
    df_affinity_sm = pd.DataFrame(aux, index=df_affinity.index, columns=df_affinity.columns) 
    masses = np.sum(array_connection, axis=0)
    cache_hash = str(hashlib.sha256(df_affinity_sm.values.tobytes()).hexdigest())
    cache_file = './caches/' + cache_hash + '.json'
    try:
        import json
        f = open(cache_file)
        semantics = json.load(f)
        f.close()
    except FileNotFoundError:        
        semantics = cm.semantics_network(df_affinity_sm, masses)
        json = json.dumps(semantics)
        f = open(cache_file,"w")
        f.write(json)
        f.close()
        #semantics.to_csv('./caches/' + cache_hash + '.csv')

    edge_list['Source'] = edge_list['Source'].str.lower()
    edge_list['Target'] = edge_list['Target'].str.lower()

    red = sa.affinity_adjacency_to_network(edge_list, sym=False, edge_list=True)
    # top_semantics = heapq.nlargest(trh, semantics)[::-1]
    # nodes_select = [x for x,y in red.nodes(data=True) if x in top_semantics]
    # red = red.subgraph(nodes_select)

    return red, semantics, edge_list, array_affinity, array_connection, df_affinity, diccionario_masas


def save_heatmap(distances, salida='distances.pdf'):
    '''
    Saves a heatmap figure using seaborn in pdf.

    :param distances: matrix.
    '''
    import matplotlib.pyplot as plt
    plt.rcParams["axes.grid"] = False
    import seaborn as sns

    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    def swap_rows(a, ix, jx):
        temp = a[ix, :].copy()
        #temp_name = a.index[ix]
        b = a[jx, :]

        a[ix, :] = b
        a[jx, :] = temp

        #a.index.names[ix] = a.index.names[jx]
        #a.index.names[jx] = temp_name

        return a
        #return a.reindex([a.index[ix], a.index[jx]])


    def swap_cols(a, ix, jx):
        temp = a[:, ix].copy()
        #temp_name = a[ix]
        b, c = a[:, ix], a[:, jx]

        a[:, ix] = c
        a[:, jx] = temp

        #a.columns[ix] = a.columns[jx]
        #a.columns[jx] = temp_name

        return a
        #return a.reindex(columns=[a.columns[ix], a.columns[jx]])

    sns.set()
    #distances = distances / distances.max().max()
    num_entities = distances.shape[0] / 10
    plt.figure(figsize = (12*num_entities,9*num_entities))
    sns.set(font_scale=1.5)

    #This is to make the plot more pretty: we make closer in the plot stuff that
    # is somewhat closer in the data (very simple algorithm, the single linkage)
    clusters = fcluster(linkage(distances.values, optimal_ordering=True), 1, criterion='distance')
    res_distances = distances.copy().values
    list_names = list(distances.columns)
    actual_col = 0
    for community in np.unique(clusters):
        cols = clusters == community

        for i in cols.nonzero()[0]:

            res_distances = swap_rows(res_distances, actual_col, i)
            res_distances = swap_cols(res_distances, actual_col, i)

            aux = list_names[i]
            list_names[i] = list_names[actual_col]
            list_names[actual_col] = aux
            #res_df.rename(columns={res_df.columns[actual_col]: res_df.columns[i], res_df.columns[i]: res_df.columns[actual_col]},
                    #index={res_df.index[actual_col]: res_df.index[i], res_df.index[i]: res_df.index[actual_col]},
                    #inplace=True)

           # res_df = res_df.rename(columns={res_df.columns[actual_col]: res_df.columns[i], res_df.columns[i]: res_df.columns[actual_col]},
           #                        index={res_df.columns[actual_col]: res_df.columns[i], res_df.columns[i]: res_df.columns[actual_col]})

            actual_col += 1
    res_df = pd.DataFrame(res_distances)
    res_df.index = list_names
    res_df.columns = list_names
    sns_plot = sns.heatmap(res_df, xticklabels=1, yticklabels=1, cmap='Reds', vmin=0, vmax=1)
    sns_plot.set_xticklabels(labels=res_df.columns, rotation=45)

    if salida is not None:
        sns_plot.figure.savefig(salida, bbox_inches='tight')

    return res_df

def known_pipe_distance(edge_list, top = 25, save=None):
    red, semantics, _, _, _, df_affinity, degrees = pipe_preliminars(edge_list, 0, preprocess=False)
    distances = compute_pipe_distances(red, df_affinity, semantics, top=top, original_degree=degrees)

    if save is not None:
        save_heatmap(distances, salida=save + '.pdf')
        distances.to_csv(save + '.csv', index=True, header=True)

    return distances

def generate_figures_from_pdf(tribe=['celts', 'greek', 'edda', 'myth', 'iliad', 'homeric', 'odyssey']):
     tops = [10, 25, 50, 100]
     for top in tops:
         distances = pd.read_csv('Pipes/' + tribe + '_' + str(top) + '.csv', index_col=0)
         save_heatmap(distances, 'Pipes/' + tribe + '_' + str(top) + '.pdf')

def main():
    import experiments as ex

    def remesa(top):
        known_pipe_distance(ex.celts, top=top, save='celts_'+str(top))
        known_pipe_distance(ex.edda, top=top, save='edda_'+str(top))
        known_pipe_distance(ex.greek, top=top, save='greek_'+str(top))

        known_pipe_distance(ex.odyssey, top=top, save='odyssey_'+str(top))
        known_pipe_distance(ex.iliad, top=top, save='iliad_'+str(top))

        known_pipe_distance(ex.myth, top=top, save='myth_'+str(top))
        known_pipe_distance(ex.homeric, top=top, save='homeric_'+str(top))

    #This might be heavy!
    tops = [10, 25, 50, 100]
    for top in tops:
        remesa(top)

def true_main(name, top):
    mythh = name
    if mythh == 'edda':
        mm = sa.canon_edda()
        entities = []
        #entities = ['Odin', 'Name', 'Thor', 'Loke', 'Hvergelmer', 'Son', 'Dwarf', 'Sigurd', 'Frey', 'Freyja' ]
    elif mythh == 'greek':
        mm = sa.canon_greek()
        entities = []
        #entities = ['Heracles', 'Theseus', 'Apollo', 'Psyche', 'Jason', 'Eurystheus', 'Zeus', 'Perseus', 'Man', 'Pelias']
    elif mythh == 'celt':        
        mm = sa.canon_celt()
        entities = []
        #entities = ['Ireland', 'Lugh', 'Balor', 'Turann', 'Conary', 'King', 'Son', 'Gobhaun', 'Earth', 'Dagda']
    elif mythh == 'myth':
        mm = sa.canon_myth_all()
        entities = []
        #entities = ['Odin', 'Thor', 'Name', 'Son', 'Heracles', 'Loke', 'Hvergelmer', 'Lugh', 'King', 'Ireland']


    if len(entities) != top:
        entities = None
    red, semantics, edge_list, array_affinity, array_connection, df_affinity, diccionario_masas = pipe_preliminars(mm, alpha=0.9)
    distances, rdf = compute_pipe_distances(red, df_affinity, semantics, original_degree=diccionario_masas, entities=entities, top=top, verbose=False, affinity_mode=True, local_af=True)
    cluster_distances = save_heatmap(distances, './Images/' + mythh + '_' + str(top) + '_distances.pdf')
    cluster_distances.to_csv(mythh + '_' + str(top) + '_ss.csv')
    
if __name__ == '__main__':
    #main()
    import sys
    #sys.argv = ['', 'edda', '10']
    true_main(sys.argv[1], int(sys.argv[2]))
    
    