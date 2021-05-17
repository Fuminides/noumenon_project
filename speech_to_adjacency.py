# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:19:36 2019

@author: Javier Fumanal Idocin

This comprises a list of helpers to load the data and transform it into a network.
I mainly work with dictionaries, edge dataframe and adjacency array/dataframe, so
the conversors between those datatypes are also implemented here.

The chosen texts are cached so that you always use the same network.
I also included Odyssey and Iliad as a bonus. I did not include them in the
final paper because the style and theology in those tales was much different
than the rest of the tales. Some additional observations could be done on them,
however, but I thought it was not worth it.

IF YOU WANT TO BUILD THE NETWORKS FROM THE PAPER:
    -Look for the functions at the end of the file. They should run even if you
    did not download any of the texts.
    
Again, I'm sorry I don't have the time to take care of this code in more detail.
If you feel like you can help finding bugs or anything else contact me without hesitation.

Cite any of my works if you use this code, please.
"""
import nltk
import re
import itertools
import pandas as pd
import numpy as np
import networkx as nx
import numbers


import matplotlib.pyplot as plt

greek = 'Greek tales'
celtic = 'Celtic wonder tales'
iliad = 'Iliad'
odyssey = 'Odyssey'
edda = 'Edda'

known_books = {celtic: ('archive', 'celticwondertale00younrich'),
               edda: ('gutenberg', 18947),
               iliad: ('gutenberg', 6130), odyssey: ('gutenberg', 1727),
               greek: ('archive', ('in.ernet.dli.2015.76946', '2015.76946.Greek-Myths'))}

from nltk.stem import PorterStemmer

SEP_CHAR = '€€'


def _order_dict(x):
    return {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}


def extract_entities(text):
    '''
    Returns a set with all the different entities found on a string.
    '''
    from nltk.tag.stanford import StanfordNERTagger
    st = StanfordNERTagger('/home/fuminides/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz',
                           '/home/fuminides/stanford-ner-2018-10-16/stanford-ner.jar')
    people = []

    token_text = nltk.word_tokenize(text)
    tagged = st.tag(token_text)
    tags = [tagged[x][1] for x in range(len(tagged))]

    for ix, tag in enumerate(tags):
        if tag == 'PERSON' and len(tagged[ix][0]) > 2:
            people.append(tagged[ix][0])

    return set(people)

def load_edges(path, sym=True):
    res = pd.read_csv(path)
    
    if sym:
        aux = pd.read_csv(path)
        for row in aux.iterrows():
            source, target, weight = row[1]
            dic_aux = {'Source': target, 'Target': source, 'Weight':weight}
            res = res.append(dic_aux, ignore_index=True)
        res = res.groupby(['Source', 'Target']).max().reset_index()
    return res

def load_known_libro(libro, delete=False):
    '''
    Loads book from the internet. It must be in the known books key list.

    '''
    import os

    global known_books
    if libro in known_books.keys():
        origen, identifier = known_books[libro]

        if origen == 'gutenberg':
            try:
                from gutenberg.acquire import load_etext
                from gutenberg.cleanup import strip_headers

                return strip_headers(load_etext(identifier)).strip()
            except ModuleNotFoundError:
                return open("Myths/gutenberg/" + libro + ".txt", "r", encoding='utf8').read()
        elif origen == 'archive':
            if len(identifier) == 2:
                identifier, file_id = identifier
            else:
                file_id = identifier
                
            if os.path.exists('./' + identifier + '/' + file_id + '_djvu.txt'):
                with open('./' + identifier + '/' + file_id + '_djvu.txt', 'r', encoding='utf8') as file:
                    data = file.read()
            else:
                import internetarchive
    
                item = internetarchive.get_item(identifier)
                item.download(file_id + '_djvu.txt')
    
                with open('./' + identifier + '/' + file_id + '_djvu.txt', 'r', encoding='utf8') as file:
                    data = file.read()

            if delete:
                os.remove('./' + identifier + '/' + file_id + '_djvu.txt')
                os.rmdir('./' + identifier + '/')

            return data
    else:
        print('Book not known')
        return None


def basic_entity_extraction(text0):
    text = text0.split(' ')

    dump_dummy(text)
    preprocess_whole_text(text)
    dump_dummy(text)

    text = purge_non_entities(text, remove_duplicates=False)
    return [x for x in text if len(x) > 0]


def entity_overlap_texts(text1, text2, text3=None):
    '''
    Text in their string form.
    '''
    text1 = text1.split(' ')
    text2 = text2.split(' ')
    if text3 is not None:
        text3 = text3.split(' ')

    dump_dummy(text1)
    preprocess_whole_text(text1)
    dump_dummy(text1)

    dump_dummy(text2)
    preprocess_whole_text(text2)
    dump_dummy(text2)

    if text3 is not None:
        dump_dummy(text3)
        preprocess_whole_text(text3)
        dump_dummy(text3)

    text1 = purge_non_entities(text1, remove_duplicates=False)
    text2 = purge_non_entities(text2, remove_duplicates=False)
    if text3 is not None:
        text3 = purge_non_entities(text3, remove_duplicates=False)

    text1 = [item for sublist in text1 for item in sublist]
    text2 = [item for sublist in text2 for item in sublist]
    if text3 is not None:
        text3 = [item for sublist in text3 for item in sublist]

    s1 = set(text1)
    s2 = set(text2)
    if text3 is not None:
        s3 = set(text3)

    hist1 = {i: text1.count(i) for i in s1}
    hist2 = {i: text2.count(i) for i in s2}
    if text3 is not None:
        hist3 = {i: text3.count(i) for i in s3}

    s12 = s1.intersection(s2)
    if text3 is not None:
        s123 = s1.intersection(s2).intersection(s3)

    from collections import Counter
    result = Counter()
    if text3 is None:
        for elem in [hist1, hist2]:
            for key, value in elem.items():
                result[key] += value
    else:
        for elem in [hist1, hist2, hist3]:
            for key, value in elem.items():
                result[key] += value

    if text3 is None:
        return len(s12) / len(s1.union(s2)), sum([result[x] for x in s12]) / sum([result[x] for x in s1.union(s2)])
    else:
        return len(s123) / len(s1.union(s2).union(s3)), sum([result[x] for x in s123]) / sum(
            [result[x] for x in s1.union(s2).union(s3)])


def basic_book_analysis(book_text, k=10):
    '''
    Performs text exploratory analysis on a string that supposdely comes from a book.
    '''
    content = book_text.replace('\n', '').split(' ')
    book_paras = book_text.split('\n')
    n_words = len(content)
    dump_dummy(book_paras)

    book_paras = preprocess_whole_text(book_paras)
    dump_dummy(book_paras)

    content_entities = purge_non_entities(book_paras, remove_duplicates=True)
    content_entities = [item for sublist in content_entities for item in sublist]
    n_entities = len(book_paras)

    hist_entities = {i: content_entities.count(i) for i in set(content_entities)}

    import difflib

    actual_words = content
    container = []
    [container.extend(re.sub('\n+', '\n', x).replace(';', '').replace(',', '').replace('.', '').split('\n')) for x in
     actual_words]
    container = list(set(container))
    keys = list(hist_entities.keys())
    for a in keys:
        try:
            hist_entities[difflib.get_close_matches(a, container, 1, 0.4)[0]] = hist_entities.pop(a)
        except KeyError:
            pass
        except IndexError:
            pass
    hist_entities = {k: v for k, v in sorted(hist_entities.items(), key=lambda item: item[1])}
    most_important = list(hist_entities)[-k:]

    return n_words, n_entities, most_important, hist_entities

def analysis_books(libros=list(known_books.keys())):
    '''
    Performs the analysis on all the known books.
    '''
    hists = []
    n_list_entities = []
    n_words = []
    importants = []

    for book in libros:
        print('Processing: ' + book)
        text = load_known_libro(book, delete=True)
        n_word, n_entities, most_important, hist_entities = basic_book_analysis(text)

        n_list_entities.append(n_entities)
        importants.append(most_important)
        hists.append(hist_entities)
        n_words.append(n_word)

    fig, ax = plt.subplots()
    plt.xlabel('n-th most common word')
    plt.ylabel('$log(Appearances)$')

    for ix, histo in enumerate(hists):
        name = libros[ix]
        ax.plot(np.arange(len(histo.values()))[::-1], np.log(np.array(list(histo.values()))), label=name)

    ax.legend(loc='upper right', shadow=False)
    plt.savefig('Global hist.pdf')

    return hists, n_list_entities, n_words, importants

def add_edges_df_clustering(dfs):
    '''
    Returns a edge-based sum of n edge format dfs, and a dict
    that of each entity, and their corresponding original df, acording to its frequency
    in each one.
    '''
    entities = set()
    df_dummy = pd.DataFrame()
    n_words = np.zeros((len(dfs)))
    for ix, df in enumerate(dfs):
        entities = entities.union(set(df['Source']).union(set(df['Target'])))
        n_words[ix] = np.sum(df['Weight'])

    frequencies = {}
    clustering = {}
    for entity in entities:
        frequencies[entity] = [0]*len(dfs)
        for ix, df in enumerate(dfs):
            frequencies[entity][ix] = (np.sum(df['Source'] == entity) + np.sum(df['Target'] == entity)) / n_words[ix]

        clustering[entity] = np.argmax(frequencies[entity])

    for df in dfs:
        df_dummy = df_dummy.append(df)

    df_dummy.groupby(['Source', 'Target'], as_index=False).sum()

    return df_dummy, clustering

def export_histogram(hist_entities, name):
    plt.figure()
    plt.plot(np.array(list(hist_entities.values())))
    plt.title('Sorted word appearances in the ' + name)
    plt.xlabel('N-ism most repeated word')
    plt.ylabel('Number of appearances')
    plt.savefig(name.replace(' ', '_') + '_word_counts.pdf')




def process_text(libro, granularidad='parrafos', empaquetar=False, encoding='cp1252'):
    '''
    Given a text in a single string, it returns it in the proper packaging to perform
    edge extraction.
    '''

    if granularidad == 'lineas':
        content = libro.split("\n")
        dump_dummy(content)
    elif granularidad == 'parrafos':
        content = libro.split("\n\n")
        dump_dummy(content)
    elif granularidad == 'palabras':
        content = libro.split(' ')
        dump_dummy(content)
    else:
        words = libro.split(' ')
        dump_dummy(words)
        content = []
        palabras = len(words)
        area = granularidad
        for i in range(area, palabras - area):
            content.append(' '.join(words[(i - area):(i + area)]))

        for i in range(palabras - area, palabras):
            content.append(' '.join(words[i:]))
    if empaquetar:
        res = []
        for linea in content:
            aux = linea.replace('Mr.', 'Mr').split('.')
            if aux[-1] == '\n':
                del aux[-1]

            res.append(aux)
        return res
    else:
        return content


def dump_dummy(text):
    '''
    Erases empty or 1-len lists from a list of lists.
    '''
    a_eliminar = []
    for ix, i in enumerate(text):
        if len(i) <= 1:
            a_eliminar.append(ix)

    a_eliminar.reverse()

    for muere in a_eliminar:
        del text[muere]


def preprocess(line):
    '''
    Tokeninzes a list of strings.
    '''
    ps = PorterStemmer()

    line = nltk.word_tokenize(line)
    line = nltk.pos_tag(line)
    line = [(ps.stem(i[0]), i[1]) for i in line]

    return line


def preprocess_whole_text(book, paquetes=False):
    '''
    '''
    lemma = nltk.wordnet.WordNetLemmatizer()
    
    if not paquetes:
        book = nltk.pos_tag_sents([nltk.word_tokenize(line) for line in book], lang='eng')
        
        for ix, line in enumerate(book):
            def line_to_lemma(line):
                res = []
                for i in line:
                    try:
                        res.append((lemma.lemmatize(i[0], i[1]), i[1]))
                    except KeyError:
                        res.append((i[0], i[1]))
                    
                return res
            book[ix] = line_to_lemma(line)
    else:
        for ix, paquete in enumerate(book):
            preprocess_whole_text(paquete, False)
    
    return book

def look_nouns(tokens):
    '''
    Returns the nouns in a tokenized list where the tagger
    said it was a noun.
    '''
    entidades = []
    prog = re.compile('NN.*')

    for (word, tag) in tokens:

        if (len(word) > 2) and prog.match(tag):
            entidades.append(word.capitalize())

    return entidades


def purge_non_entities(token_text, paquetes=False, remove_duplicates=True):
    '''
    Given a tagged text, removes the non noun tagged words.
    '''
    result = []

    if not paquetes:
        for ix, line in enumerate(token_text):
            entities = look_nouns(line)
            if not remove_duplicates:
                result.append(entities)
            else:
                try:
                    if line[0][0] == entities[0]:
                        result.append(entities)
                except IndexError:
                    pass

    else:
        for ix, paquete in enumerate(token_text):
            result.append(purge_non_entities(paquete, False))
    return result


def _flatten_list(l):
    return [item for sublist in l for item in sublist]


def entities_to_adjacency_graph(frases, paquetes=True, granularidad='parrafos'):
    '''
    '''
    global SEP_CHAR

    adjacency_graph = {}
    if not paquetes:
        for ix, element in enumerate(frases):
            #element = element.split(' ')
            if not isinstance(granularidad, numbers.Number):
                combinaciones = list(itertools.combinations(element, 2))
                
                for combinacion in combinaciones:
                    
                    combinacion = sorted(combinacion)
                    clave = combinacion[0].replace('\'', '') + SEP_CHAR + combinacion[1].replace('\'', '')
                    try:
                        adjacency_graph[clave] += 1
                    except KeyError:
                        adjacency_graph[clave] = 1
            else:
                last_word = element[0]
                for word_index, word in enumerate(element):
                    clave = last_word + SEP_CHAR + word
                    try:
                        adjacency_graph[clave] += 1
                    except KeyError:
                        adjacency_graph[clave] = 1
    else:
        # frases = frases[0]
        for ix, frase in enumerate(frases):
            for jx, element in enumerate(frase):
                combinaciones = list(itertools.combinations(element, 2))
                for combinacion in combinaciones:
                    combinacion = sorted(combinacion)
                    clave = combinacion[0].replace('\'', '') + SEP_CHAR + combinacion[1].replace('\'', '')
                    try:
                        adjacency_graph[clave] += 1
                    except KeyError:
                        adjacency_graph[clave] = 1
    return adjacency_graph


def pairkeys_to_array(dictionary_pair):
    '''
    '''
    global SEP_CHAR

    unique_keys = {}

    for key in dictionary_pair.keys():
        (a, b) = key.split(SEP_CHAR)
        try:
            unique_keys[a] += 1
        except KeyError:
            unique_keys[a] = 1
        try:
            unique_keys[b] += 1
        except KeyError:
            unique_keys[b] = 1

    final_keys = sorted(list(unique_keys.keys()))
    result = pd.DataFrame(np.identity(len(final_keys)), columns=final_keys)
    # result = result.set_index(final_keys) #Soo slow!!
    result.index = final_keys

    for i in dictionary_pair.items():
        (key, value) = i
        (key1, key2) = key.split(SEP_CHAR)
        result[key1][key2] = value
        result[key2][key1] = value

    return result


def data_frame_array_named(data):
    '''
    '''
    values = data.values
    good_dict = {}
    reverse_dict = {}

    for ix, element in enumerate(list(data)):
        good_dict[element] = ix
        reverse_dict[ix] = element

    return values, good_dict, reverse_dict


def adjacency_to_csv(adj_dictionary, name_file='./new_dataset.csv', graph_commons=False):
    '''
    '''
    global SEP_CHAR

    sorted_tuples = sorted(adj_dictionary.items(), key=lambda x: x[1], reverse=True)
    if graph_commons:
        file = open(name_file, mode='w', encoding='utf8')
        file.write('FromType,FromName,Edge,ToType,ToName,Weight\n')

        for elem in sorted_tuples:
            (a, b) = elem[0].split(SEP_CHAR)
            # a = a.encode('utf8');b = b.encode('utf8')
            if a != b:
                file.write('Word,' + a + ', APPEARS, Word,' + b + ',' + str(elem[1]) + '\n')

    else:
        file = open(name_file, mode='w', encoding='utf8')
        file.write('Source;Target;Weight\n')
        # Iterate over the sorted sequence
        for elem in sorted_tuples:
            (a, b) = elem[0].split(SEP_CHAR)
            # a = a.encode('utf8');b = b.encode('utf8')

            file.write(a + ';' + b + ';' + str(elem[1]) + '\n')

    file.close()


def adjacency_dict_to_edges_df(adj_dictionary):
    '''
    '''
    global SEP_CHAR
    sorted_tuples = sorted(adj_dictionary.items(), key=lambda x: x[1], reverse=True)
    res = pd.DataFrame(columns=['Source', 'Target', 'Weight'])
    # Iterate over the sorted sequence
    for ix, elem in enumerate(sorted_tuples):
        (a, b) = elem[0].split(SEP_CHAR)
        # a = a.encode('utf8');b = b.encode('utf8')
        res.loc[ix] = (a, b, elem[1])

    return res


def clustering_to_csv(diccionario, clustering, name='./new_dataset_clustering.csv'):
    '''
    '''
    file = open(name, mode='w', encoding='utf8')
    order = sorted(range(len(diccionario.keys())), key=lambda k: list(diccionario.keys())[k].lower())
    file.write('Id;Label;Clustering\n')
    for ix in order:
        file.write(list(diccionario.keys())[ix] + ';' + list(diccionario.keys())[ix] + ';' + str(clustering[ix]) + '\n')

    file.close()


def cut_dict(dic, threshold):
    '''
    '''
    return {k: v for k, v in dic.items() if v > threshold}


def cut_edges(df, threshold):
    '''
    '''
    indices = [x for x in range(df.shape[0]) if df.iloc[x]['Weight'] > threshold]
    return df.loc[indices, :]


def data_frame_to_adj_dict(df):
    '''
    '''
    global SEP_CHAR

    columns_names = list(df)
    row_names = list(df.index)
    res = {}

    for ix, col in enumerate(columns_names):
        for ij, row in enumerate(row_names):
            if df.iloc[ix, ij] > 0:
                res[str(col) + SEP_CHAR + str(row)] = df.iloc[ix, ij]

    return res

def adjacency_df_to_edges_df(df):
    '''
    DF of type: Adjacency matrix to (Source, Target -> Weight) type DF.
    '''
    return adjacency_dict_to_edges_df(data_frame_to_adj_dict(df))

def edges_df_to_adjacency_df(df, symmetry=False):
    '''
    (Source, Target -> Weight) type DF to DF of type: Adjacency matrix.
    '''
    return edges2adjacency_df(df, symmetry=symmetry)

def data_frame_format_networkx(df):
    '''
    Adapts an affinity dataframe to a edges dataframe.
    '''
    col_names = list(df)
    row_names = list(df.index)
    result = pd.DataFrame(columns=['Source', 'Target', 'Weight'])

    for ix, row in enumerate(row_names):
        for ij, col in enumerate(col_names):
            if df.iloc[ix, ij] > 0:
                aux = pd.DataFrame([(row, col, df.iloc[ix, ij])], columns=['Source', 'Target', 'Weight'])
                result = pd.concat([result, aux], ignore_index=True)

    return result


def affinity_adjacency_to_network(adjacency_list_affinity, sym=False, edge_list=True):
    if sym:
        graph_type = None
    else:
        graph_type = nx.DiGraph()

    if edge_list and hasattr(nx, 'from_pandas_dataframe'):
        return nx.from_pandas_dataframe(adjacency_list_affinity, 'Source', 'Target', ['Weight'],
                                        create_using=graph_type)
    elif edge_list and hasattr(nx, 'from_pandas_edgelist'):
        return nx.from_pandas_edgelist(adjacency_list_affinity, 'Source', 'Target', ['Weight'], create_using=graph_type)
    else:  # edge_list False
        return nx.from_pandas_adjacency(adjacency_list_affinity, create_using=graph_type)
    
def filter_uninportant(people, histogram, min_appears=5):
    '''
    Filter the elements in people using the correspondent histogram.

    :parameter people: list of people/elements.
    :parameter histogram: dictionary of element -> appearances.
    '''
    vip = []
    for x in people:
        try:
            if histogram[x] > min_appears: vip.append(x)
        except KeyError:
            pass

    return vip


def text2edge_list(heart, thr=0, reconstruct=False, f_affinity=None, granularidad='parrafos'):
    original_heart = heart
    if granularidad == 'parrafos':
        heart = heart.split('\n\n')

    preprocess_whole_text(heart)
    dump_dummy(heart)

    heart = purge_non_entities(heart, remove_duplicates=True)
    adj_dict = entities_to_adjacency_graph(heart, False, granularidad)

    adj_dict = cut_dict(adj_dict, threshold=thr)

    if f_affinity is not None:
        entidades = np.unique(
            [x.split(SEP_CHAR)[0] for x in adj_dict.keys()] + [x.split(SEP_CHAR)[1] for x in adj_dict.keys()])
        num_entidades = len(entidades)
        A = np.zeros([num_entidades, num_entidades], np.float64)
        A = pd.DataFrame(A)
        A.columns = entidades
        A.index = entidades
        # Construct the adjacency pandas
        for key, peso in adj_dict.items():
            # (key, peso) = elem
            (source, target) = key.split(SEP_CHAR)

            A[source][target] = peso
            A[target][source] = peso

        data = f_affinity(A.values)
        A[:] = data

    if reconstruct:
        import difflib

        actual_words = original_heart
        container = []
        [container.extend(re.sub('\n+', '\n', x).replace(';', '').replace(',', '').replace('.', '').split('\n')) for x
         in actual_words]
        container = list(set(container))
        aux_dict = {}
        for pair_keys in adj_dict.keys():
            (a, b) = pair_keys.split(SEP_CHAR)
            if len(a) > 1 and len(b) > 1:
                k1 = difflib.get_close_matches(a, container, 1, 0.4)[0]
                k2 = difflib.get_close_matches(b, container, 1, 0.4)[0]

                aux_dict[k1 + SEP_CHAR + k2] = adj_dict[pair_keys]

        adj_dict = aux_dict

    return adjacency_dict_to_edges_df(adj_dict)


def crop_bad_edges(net):
    global SEP_CHAR
    forbidden_words = ['the', 'The', 'and', 'And']

    if isinstance(net, dict):
        res = {}
        for key, elem in net.items():
            source, target = key.split(SEP_CHAR)
            if (source not in forbidden_words) and (target not in forbidden_words):
                if (source != target) and (len(source) > 1) and (len(target) > 1):
                    try:
                        res[source.capitalize() + SEP_CHAR + target.capitalize()] += elem
                    except KeyError:
                        res[source.capitalize() + SEP_CHAR + target.capitalize()] = elem

        return res
    else:
        net = net[net['Source'] != net['Target']]
        net = net[net['Source'].apply(len) > 1]
        net = net[net['Target'].apply(len) > 1]
        net['Source'] = net['Source'].apply(lambda a: re.sub(r'\W+', '', a))

        net['Target'] = net['Target'].apply(lambda a: re.sub(r'\W+', '', a))

        net = net.loc[net['Target'] not in forbidden_words]
        net = net.loc[net['Source'] not in forbidden_words]

        return net


def full_process(path_libro=None, out=None, thr=0, reconstruct=False, f_affinity=None, granularidad='parrafos', encoding='utf8', heart=None):
    if path_libro is not None:
        original = open(path_libro, 'r').read()
        heart = process_text(original, granularidad, encoding=encoding)
    else:
        actual_words = set([item for sublist in heart for item in sublist])
        
    if granularidad == 'parrafos':
        dump_dummy(heart)

    heart = preprocess_whole_text(heart)
    dump_dummy(heart)

    heart = purge_non_entities(heart, remove_duplicates=True)
    adj_dict = entities_to_adjacency_graph(heart, False, granularidad)
    adj_dict = crop_bad_edges(adj_dict)
    adj_dict = cut_dict(adj_dict, threshold=thr)

    if f_affinity is not None:
        entidades = np.unique(
            [x.split(SEP_CHAR)[0] for x in adj_dict.keys()] + [x.split(SEP_CHAR)[1] for x in adj_dict.keys()])
        num_entidades = len(entidades)
        A = np.zeros([num_entidades, num_entidades], np.float64)
        A = pd.DataFrame(A)
        A.columns = entidades
        A.index = entidades
        # Construct the adjacency pandas
        for key, peso in adj_dict.items():
            # (key, peso) = elem
            (source, target) = key.split(SEP_CHAR)

            A[source][target] = peso
            A[target][source] = peso

        data = f_affinity(A.values)
        A[:] = data
        adj_dict = data_frame_to_adj_dict(A)
        adj_dict = crop_bad_edges(adj_dict)

    if reconstruct:
        import difflib
        if path_libro is not None:
            original = set(open(path_libro, 'r').read().split(' '))
            actual_words = process_text(original, 'palabras', encoding=encoding)
            
        container = []
        [container.extend(re.sub('\n+', '\n', x).replace(';', '').replace(',', '').replace('.', '').split('\n')) for x
         in actual_words]
        container = list(set(container))
        aux_dict = {}
        for pair_keys in adj_dict.keys():
            (a, b) = pair_keys.split(SEP_CHAR)
            if len(a) > 1 and len(b) > 1:
                try:
                    k1 = difflib.get_close_matches(a, container, 1, 0.4)[0]
                except IndexError:
                    k1 = a
                try:
                    k2 = difflib.get_close_matches(b, container, 1, 0.4)[0]
                except IndexError:
                    k2 = b

                aux_dict[k1 + SEP_CHAR + k2] = adj_dict[pair_keys]

        adj_dict = aux_dict

   
    adjacency_data_frame = pairkeys_to_array(adj_dict)
    (adjacency_matrix, dictionary_entity, dictionary_reverse) = data_frame_array_named(adjacency_data_frame)

    if out is not None:
        adjacency_to_csv(adj_dict, out)
        print('Chapter saved')

    return adj_dict

def _comb_affinity(a, alpha=0.5):
    import affinity as af
    from numba import cuda
    try:
        cuda.select_device(0)
        return af.conex2af_GPU_std_comb(a, alpha)
    except cuda.cudadrv.error.CudaSupportError:
        return af.conex2af_std_comb(a, alpha)

def generate_known_network_dict(name, granularidad=10, aff=_comb_affinity, thr=10):
    '''
    Only on known books.
    
    Generates a network given a granularity, an affinity function and a edge threshold.
    '''
    text = load_known_libro(name)
    text_pack = process_text(text, granularidad=granularidad)
    
    return full_process(heart=text_pack, granularidad=granularidad, reconstruct=False, f_affinity=aff, thr=thr)

def semantics_dict_csv(semantics, name_file='./data_set_semantics.csv'):
    file = open(name_file, mode='w', encoding='utf8')
    file.write('Id;Semantic\n')

    for key, value in semantics.items():
        file.write(str(key) + ';' + str(value) + '\n')


def generate_histogram(path_libro, encoding='utf8'):
    '''
    Given the path to a text return the 1-gram histogram.
    '''
    libro = open(path_libro, 'r', encoding=encoding).read()
    heart = process_text(libro, granularidad='parrafos', encoding=encoding)
    dump_dummy(heart)

    preprocess_whole_text(heart)
    dump_dummy(heart)

    heart = purge_non_entities(heart, remove_duplicates=True)
    heart = _flatten_list(heart)
    dump_dummy(heart)

    hist_entities = {i: heart.count(i) for i in set(heart)}
    return {k: v for k, v in sorted(hist_entities.items(), key=lambda item: item[1])}

def word_filter(text, allowed):
    res = ''
    for word in text.split(' '):
        if word in allowed:
            res += ' ' + word
    
    return res

def edges2adjacency_df(edges_df,symmetry=False):
    '''
    Returns the adjacency version of an edges df.
    '''
    unicos = set(edges_df['Source']).union(set(edges_df['Target']))

    res = pd.DataFrame(np.zeros((len(unicos), len(unicos))))
    res.columns = unicos
    res.index = unicos

    for ix, elem_row in enumerate(edges_df.iterrows()):
        try:
            source, target, weight = elem_row[1]
        except ValueError:
            source, target = elem_row[1]
            weight = 1
            
        res[source][target] = weight
        if symmetry:
            res[target][source] = weight

    return res

def filter_top_nodes(adjacency_df, top=1000, normalize=False):
    if len(adjacency_df.columns) == 3:
        inverse = True
        aux = adjacency_df
        adjacency_df = edges2adjacency_df(adjacency_df, True)
    else:
        inverse = False
        
    degrees = np.sum(adjacency_df)
    arguments = degrees.argsort()[-top:][::-1]
    
    nodes = adjacency_df.columns[arguments]
    
    if normalize:
        norm_factor = np.max(np.max(adjacency_df.loc[nodes][nodes]))
    else:
        norm_factor = 1.0
        
    if inverse:
        filtered = np.array([aux['Source'].iloc[x] in nodes for x in range(len(aux['Source']))]) *  np.array([aux['Target'].iloc[x] in nodes for x in range(len(aux['Target']))])
        chosen = aux.loc[filtered,:] 
        chosen['Weight'] = chosen['Weight'] / norm_factor
        return chosen, nodes
    else:
        return adjacency_df.loc[nodes][nodes] / norm_factor, nodes

# =============================================================================
# EXPERIMENTS: FUNCTIONS TO REPLICATE THE EXPERIMENTS
# =============================================================================
### EXPERIMENTS: FUNCTIONS TO REPLICATE THE EXPERIMENTS 
# (HEADER FOR SPYDER)
# =============================================================================
# GENERATE THE NETWORKS
# =============================================================================
def canon_edda():
    cache_file = 'edda.csv'
    try:
        edda_df0 = pd.read_csv(cache_file, header=0, index_col=0)
    except FileNotFoundError:    
        edda_net = generate_known_network_dict(edda, aff=None, thr=1)
        edda_df = adjacency_dict_to_edges_df(edda_net)
        edda_df0, _ = filter_top_nodes(edda_df, 300, normalize=False)
        edda_df0.to_csv(cache_file)
        
    return edda_df0

def canon_celt():
    cache_file = 'celt.csv'
    try:
        celt_df0 = pd.read_csv(cache_file, header=0, index_col=0)
    except FileNotFoundError:    
        celtic_net = generate_known_network_dict(celtic, aff=None, thr=1)
        celt_df = adjacency_dict_to_edges_df(celtic_net)
        celt_df0, _ = filter_top_nodes(celt_df, 300, normalize=False)
        celt_df0.to_csv(cache_file)

    return celt_df0

def canon_greek():
    cache_file = 'greek.csv'
    try:
        greek_df0 = pd.read_csv(cache_file, header=0, index_col=0)
    except FileNotFoundError:    
        greek_net = generate_known_network_dict(greek, aff=None, thr=1)
        greek_df = adjacency_dict_to_edges_df(greek_net)
        greek_df0, _ = filter_top_nodes(greek_df, 300, normalize=False)
        greek_df0.to_csv(cache_file)
        
    return greek_df0

def canon_myth_all():
    cache_file = 'myth_all.csv'
    try:
        homeric_df = pd.read_csv(cache_file, header=0, index_col=0)
    except FileNotFoundError:
        edda_net = generate_known_network_dict(edda, aff=None, thr=1)
        edda_df = adjacency_dict_to_edges_df(edda_net)
        edda_df0, _ = filter_top_nodes(edda_df, 300, normalize=False)

        greek_net = generate_known_network_dict(greek, aff=None, thr=1)
        greek_df = adjacency_dict_to_edges_df(greek_net)
        greek_df0, _ = filter_top_nodes(greek_df, 300, normalize=False)

        celtic_net = generate_known_network_dict(celtic, aff=None, thr=1)
        celt_df = adjacency_dict_to_edges_df(celtic_net)
        celt_df0, _ = filter_top_nodes(celt_df, 300, normalize=False)

        homeric_df, _ = add_edges_df_clustering([celt_df0, edda_df0, greek_df0])
        homeric_df.to_csv(cache_file)

    return homeric_df

# =============================================================================
# COMPUTE SEMANTIC VALUE TABLES
# =============================================================================
def extract_semantic_value_all():
    #For clustering
    edda_net = generate_known_network_dict(edda, aff=None, thr=1)
    edda_df = adjacency_dict_to_edges_df(edda_net)
    edda_df0, _ = filter_top_nodes(edda_df, 300, normalize=False)
    
    greek_net = generate_known_network_dict(greek, aff=None, thr=1)
    greek_df = adjacency_dict_to_edges_df(greek_net)
    greek_df0, _ = filter_top_nodes(greek_df, 300, normalize=False)
    
    celtic_net = generate_known_network_dict(celtic, aff=None, thr=1)
    celt_df = adjacency_dict_to_edges_df(celtic_net)
    celt_df0, _ = filter_top_nodes(celt_df, 300, normalize=False)    
     
    homeric_df, _ = add_edges_df_clustering([celt_df0, edda_df0, greek_df0])
    
    import affinity as af
    import complexity as cm

    
    edda_df = edges2adjacency_df(homeric_df, symmetry=True)
    aff_net = edda_df.copy()
    aff_net[:] = af.conex2af_std_comb(aff_net.values)
    masses = np.array(edda_df.sum(axis=1))
    
    semantic_values = cm.semantics_network(aff_net, masses)
    
    return semantic_values, masses

def extract_semantic_value_table(libro):
    import affinity as af
    import complexity as cm
    
    edda_net = generate_known_network_dict(libro, aff=None, thr=1)
    edda_df = adjacency_dict_to_edges_df(edda_net)
    edda_df0, _ = filter_top_nodes(edda_df, 300, normalize=False)
    
    edda_df = edges2adjacency_df(edda_df0, symmetry=True)
    aff_net = edda_df.copy()
    aff_net[:] = af.conex2af_std_comb(aff_net.values)
    masses = np.array(edda_df.sum(axis=1))
    
    semantic_values = cm.semantics_network(aff_net, masses)
    
    return semantic_values, masses
    
    

### GENERIC FUNCTION TO ANY KNOWN BOOK AND AFFINITY FUNCTION
def process_final_network_graph(known_name):
    celtic = generate_known_network_dict(known_name, aff=None, thr=1)
    celt_df = adjacency_dict_to_edges_df(celtic)
    celt_df, _ = filter_top_nodes(celt_df, 300)
    celt_df.to_csv('./edge_lists/' + known_name + '.csv', index=None)
    generate_graph_commons(celt_df, known_name)
    
def generate_full_homeric_network():
    iliad_net = generate_known_network_dict(iliad, aff=None, thr=1)
    iliad_df = adjacency_dict_to_edges_df(iliad_net)
    iliad_df, _ = filter_top_nodes(iliad_df, 300)
    
    odyssey_net = generate_known_network_dict(odyssey, aff=None, thr=1)
    odyssey_df = adjacency_dict_to_edges_df(odyssey_net)
    odyssey_df, _ = filter_top_nodes(odyssey_df, 300)
    
    homeric_df, clustering = add_edges_df_clustering([odyssey_df, iliad_df])
    homeric_df.to_csv('./edge_lists/homeric_network.csv', index=None)
    generate_graph_commons(homeric_df, 'Homeric network', clustering)
    
    return homeric_df
    
def generate_full_myth_network(normalize=False):
    #For clustering
    edda_net = generate_known_network_dict(edda, aff=None, thr=1)
    edda_df = adjacency_dict_to_edges_df(edda_net)
    edda_df0, _ = filter_top_nodes(edda_df, 300, normalize=False)
    
    greek_net = generate_known_network_dict(greek, aff=None, thr=1)
    greek_df = adjacency_dict_to_edges_df(greek_net)
    greek_df0, _ = filter_top_nodes(greek_df, 300, normalize=False)
    
    celtic_net = generate_known_network_dict(celtic, aff=None, thr=1)
    celt_df = adjacency_dict_to_edges_df(celtic_net)
    celt_df0, _ = filter_top_nodes(celt_df, 300, normalize=False)
    _, clustering = add_edges_df_clustering([celt_df, edda_df, greek_df])
    
    #For graphics
    edda_net = generate_known_network_dict(edda, aff=None, thr=1)
    edda_df = adjacency_dict_to_edges_df(edda_net)
    edda_df, _ = filter_top_nodes(edda_df, 300, normalize=normalize)
    
    greek_net = generate_known_network_dict(greek, aff=None, thr=1)
    greek_df = adjacency_dict_to_edges_df(greek_net)
    greek_df, _ = filter_top_nodes(greek_df, 300, normalize=normalize)
    
    celtic_net = generate_known_network_dict(celtic, aff=None, thr=1)
    celt_df = adjacency_dict_to_edges_df(celtic_net)
    celt_df, _ = filter_top_nodes(celt_df, 300, normalize=normalize)
    
    homeric_df, _ = add_edges_df_clustering([celt_df, edda_df, greek_df])
    homeric_df.to_csv('./edge_lists/myth_network.csv', index=None)
    if normalize:
        generate_graph_commons(homeric_df, 'Mythic network normalized', clustering)
    else:
        generate_graph_commons(homeric_df, 'Mythic network', clustering)
    
    


#### SEND THE GRAPH TO VISUALIZE IN GRAPH COMMONS
def generate_graph_commons(df, name, clustering=None, my_key=''):
    from graphcommons import GraphCommons
    from graphcommons import Signal

    gr_sess = GraphCommons(my_key)

    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)

    nodes_created = {}
    signals = []
    '''
    #Generate nodes
    names = df.columns
    for actor in names:
        a_aux =  Signal(
                action="node_create",
                name=actor,
                type="Word",
                description=""
            )
        signals.append(a_aux)
    '''
    # Generate edges
    for row in df.itertuples():
        origen = row[1]
        origen = origen.capitalize()
        destino = row[2]
        destino = destino.capitalize()
        peso = row[3]
        if clustering is None:
            '''
            import wikipedia
            try:
                nodes_created[destino]
            except KeyError:
                try:
                    possible_images = wikipedia.page(destino).images
                    if len(possible_images) > 0:
                        signal = Signal(
                                action="node_create",
                                name = destino,
                                type = destino,
                                image = possible_images[np.argmax([destino in possible_images[x] for x in range(len(possible_images))])])
                        signals.append(signal)
                except (wikipedia.PageError, wikipedia.DisambiguationError):
                    pass
            try:
                nodes_created[origen]
            except KeyError:
                try:
                    possible_images = wikipedia.page(origen).images
                    if len(possible_images) > 0:
                        signal = Signal(
                                action="node_create",
                                name = origen,
                                type = origen,
                                image = possible_images[np.argmax([origen in possible_images[x] for x in range(len(possible_images))])])
                        signals.append(signal)
                except (wikipedia.PageError, wikipedia.DisambiguationError):
                    pass
            '''
            e_aux = Signal(
                action="edge_create",
                from_name=origen,
                from_type='ENTITY',
                to_name=destino,
                to_type='ENTITY',
                name="AFFINITY",
                weight=peso
            )
        else:

            e_aux = Signal(
                action="edge_create",
                from_name=origen,
                from_type=str(clustering[origen]),
                to_name=destino,
                to_type=str(clustering[destino]),
                name="AFFINITY",
                weight=peso
            )
        signals.append(e_aux)

    graph = gr_sess.new_graph(
        name=name,
        description="Generated from Python API wrapper",
        signals=signals
    )



