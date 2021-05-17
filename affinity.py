# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:12:50 2019

@author: Javier Fumanal Idocin.

Affinity functions in python and numba (numba might be unstable)
I know the documentation is a little bit scarce, but I hope that the names and the
code is simple enough to be understood.

Please, cite any of my works if yo use this code (the more the better hehe)
"""

# =============================================================================
# ~Affinity calculation
# =============================================================================
import numpy as np
import pandas as pd
import math

GPU_FLAG = False #If true tries to import numba.
if GPU_FLAG:
    from numba import cuda, njit, int64, float64

    # NUMBA ACCELERATED AFFINITIES
    @cuda.jit
    def _reduce_sum_2d_matrix(A, C):
        """
        Perform row reduction with sum in 2d matrix.
        """
        i = cuda.grid(1)
        if i < A.shape[0]:
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[i, k]
            C[i] = tmp
    
    @cuda.jit
    def _reduce_max_2d_matrix(A, C):
        """
        Perform row reduction with sum in 2d matrix.
        """
        i = cuda.grid(1)
        if i < A.shape[0]:
            tmp = 0.
            for k in range(A.shape[1]):
                tmp = max(tmp, A[i, k])
            C[i] = tmp
    
    @cuda.jit
    def _convex_combination(A, B, C, alpha):
        x, y = cuda.grid(2)
    
        if x < A.shape[0] and y < A.shape[1] and x != y:
            C[x,y] = alpha[0] * A[x,y] + (1-alpha[0]) * B[x,y]
    
    
    #CPU NUMBA
    @njit(float64[:,:](int64[:,:]))
    def connexion2affinity_important_friend_NB(conex):
        sumatorios = np.sum(conex, axis=1).reshape(conex.shape[0], 1)
     
        return conex / sumatorios
    
    @njit(float64[:,:](int64[:,:]))
    def connexion2affinity_best_common_friend_NB(conex):
        sumatorios = np.sum(conex, axis=1).reshape(conex.shape[0], 1)
        rows, cols = conex.shape
        res = np.zeros(conex.shape, dtype=float64)
    
        for x in range(rows):
            cx = conex[x, :]
            sx = sumatorios[x]
            for y in range(cols):
                cy = conex[y, :]
                aux = np.max(np.minimum(cx, cy)) / sx
                res[x, y] = aux[0]
    
        return res
    
    @njit(float64[:,:](int64[:,:], float64))
    def conex2af_std_comb_NB(conex, alpha=0.5):
        sumatorios = np.sum(conex, axis=1).reshape(conex.shape[0], 1)
        res = np.zeros(conex.shape, dtype=float64)
        rows, cols = conex.shape
    
        for x in range(rows):
            cx = conex[x, :]
            sx = sumatorios[x]
            for y in range(cols):
                cy = conex[y, :]
                aux = np.max(np.minimum(cx, cy)) / sx
                res[x, y] = aux[0]
    
        res2 = conex / sumatorios
    
        return res * alpha + (1-alpha) * res2
    
    #GPU Accelerated
    @cuda.jit
    def _connexion2affinity_important_friend_GPU(conex, sumatorios):
        x, y = cuda.grid(2)
        
        if x < conex.shape[0] and y < conex.shape[1]:
          conex[x, y] = conex[x, y] / sumatorios[x]
    
    @cuda.jit
    def _connexion2affinity_best_common_friend_GPU(conex, sumatorios):
        x, y = cuda.grid(2)
        
        if x < conex.shape[0] and y < conex.shape[1] and x != y:
          aux = 0
          for tmp in range(conex.shape[0]):
            aux = max(aux, min(conex[x, tmp], conex[y, tmp]))
    
          conex[x, y] = aux / sumatorios[x]
    
    
    def conex2af_GPU_important_friend(conex):
        res = conex.copy()*1.0
        threadsperblock = 32
        blockspergrid = (res.size + (threadsperblock - 1)) // threadsperblock
    
        threadsperblock2d = (16, 16)
        blockspergrid_x = math.ceil(res.shape[0] / threadsperblock2d[0])
        blockspergrid_y = math.ceil(res.shape[1] / threadsperblock2d[1])
        blockspergrid2d = (blockspergrid_x, blockspergrid_y)
    
        sumatorios_GPU = np.zeros((res.shape[0],))
        d_sum = cuda.to_device(sumatorios_GPU)
        d_res = cuda.to_device(res)
    
        _reduce_sum_2d_matrix[blockspergrid, threadsperblock](d_res, d_sum)
        _connexion2affinity_important_friend_GPU[blockspergrid2d, threadsperblock2d](d_res, d_sum)
    
        return d_res.copy_to_host()
    
    def conex2af_GPU_best_common_friend(conex):
    	res = conex.copy()*1.0
    	threadsperblock = 32
    	blockspergrid = (res.size + (threadsperblock - 1)) // threadsperblock
    
    	threadsperblock2d = (16, 16)
    	blockspergrid_x = math.ceil(res.shape[0] / threadsperblock2d[0])
    	blockspergrid_y = math.ceil(res.shape[1] / threadsperblock2d[1])
    	blockspergrid2d = (blockspergrid_x, blockspergrid_y)
    
    	sumatorios_GPU = np.zeros((res.shape[0],))
    	d_sum = cuda.to_device(sumatorios_GPU)
    	d_res = cuda.to_device(res)
    
    	_reduce_sum_2d_matrix[blockspergrid, threadsperblock](d_res, d_sum)
    	_connexion2affinity_best_common_friend_GPU[blockspergrid2d, threadsperblock2d](d_res, d_sum)
    
    	return d_res.copy_to_host()
    
    
    def conex2af_GPU_std_comb(conex,alpha=[0.5]):
    	res = conex.copy()*1.0
    	res2 = conex.copy()*1.0
    	comb0 = np.zeros(conex.shape)
    	d_alpha = cuda.to_device(np.array(alpha))
    	threadsperblock = 32
    	blockspergrid = (res.size + (threadsperblock - 1)) // threadsperblock
    
    	threadsperblock2d = (16, 16)
    	blockspergrid_x = math.ceil(res.shape[0] / threadsperblock2d[0])
    	blockspergrid_y = math.ceil(res.shape[1] / threadsperblock2d[1])
    	blockspergrid2d = (blockspergrid_x, blockspergrid_y)
    
    	sumatorios_GPU = np.zeros((res.shape[0],))
    	d_sum = cuda.to_device(sumatorios_GPU)
    	d_res = cuda.to_device(res)
    	d_res2 = cuda.to_device(res2)
    	comb = cuda.to_device(comb0)
    
    	_reduce_sum_2d_matrix[blockspergrid, threadsperblock](d_res, d_sum)
    	_connexion2affinity_best_common_friend_GPU[blockspergrid2d, threadsperblock2d](d_res, d_sum)
    	_connexion2affinity_important_friend_GPU[blockspergrid2d, threadsperblock2d](d_res2, d_sum)
    
    	_convex_combination[blockspergrid2d, threadsperblock2d](d_res, d_res2, comb, d_alpha)
    
    	return comb.copy_to_host()

def affinity_most_important_ally(list1, list2):
    '''
    Calculates the affinity based on the % of common elements between to entities.
    '''
    def common_elements(list1, list2):
        result = []

        for i in range(len(list1)):
            result.append(min(list1[i], list2[i]))

        return result


    res = common_elements(list1, list2)

    if len(res) == 0:
        return 0,0
    else:
        return np.max(res)/np.sum(list1), np.max(res)/np.sum(list2)

def connexion2affinity_network(conex):
    '''
    Given a connectivity matrix calculates the social networking affinity.
    '''
    afinidad_base = connexion2affinity_important_friend(conex)
    rows, cols = conex.shape
    res = np.zeros(conex.shape)
    
    for i in range(rows):
        suj = afinidad_base[i,:]
        for j in range(cols):
            if i != j:
                suj2 = conex[j,:]
                afinidad1, afinidad2 = affinity_networking(suj, suj2, afinidad_base, i, j) #- 2 * conex[i,j] / cols
                res[i, j] = afinidad1
                res[j, i] = afinidad2


    return res

def connexion2affinity_important_friend(conex0, csr = False):
    '''
    Given a connectivity matrix calculates the important friend affinity.
    '''
    import pandas as pd

    if isinstance(conex0, pd.DataFrame):
        conex = np.array(conex0)
    else:
        conex = conex0

    sumatorios = np.sum(conex, axis=1).reshape(conex.shape[0], 1)
 
    res =  conex / sumatorios

    np.fill_diagonal(res, 1)

    if isinstance(conex0, pd.DataFrame):
        res = pd.DataFrame(res)
        res.index = conex0.index
        res.columns = conex0.columns
    
    
    return  res

def connexion2affinity_best_common_friend(conex0):
    if isinstance(conex0, pd.DataFrame):
        conex = np.array(conex0)
    else:
        conex = conex0
        
    sumatorios = np.sum(conex, axis=1)
    res = np.zeros(conex.shape)

    for x in range(conex.shape[0]):
        for y in range(conex.shape[1]):
            res[x, y] = np.max(np.minimum(conex[x, :], conex[y, :])) / sumatorios[x]
    
    np.fill_diagonal(res, 1)
    
    if isinstance(conex0, pd.DataFrame):
        res = pd.DataFrame(res)
        res.index = conex0.index
        res.columns = conex0.columns
        
    return res

def affinity_friends(row, row2, i , j):
    '''
    Given to entities, calculates the affinity based on the most important difference
    between these two.
    '''        
    a1 = (np.abs(row - row2))
    index = np.maximum(row, row2)!=0
    a1 = 1 - a1[index] / np.maximum(row, row2)[index]
    a1 = np.mean(a1)
    return a1, a1

def affinity_basic(row, row2):
    '''
    Euclidean distances between to entities.
    '''        
    a1 = np.sum(np.abs(row - row2))
    
    return a1, a1

def affinity_enemy(row, row2):
    '''
    Return the most different affinity.
    '''        
    a1 = np.max(np.abs(row - row2))
    
    return a1, a1

def affinity_ally(list1, list2):
    '''
    Calculates the affinity based on the % of common elements between to entities.
    '''
    def common_elements(list1, list2):
        result = []
        
        for i in range(len(list1)):
            result.append(min(list1[i], list2[i]))
                
        return result


    res = common_elements(list1, list2)

    if len(res) == 0:
        return 0,0
    else:
        return np.sum(res) / np.sum(list1) , np.sum(res) / np.sum(list2)

        
def affinity_important_friend(row, row2, i, j):
    '''
    Calculates the affinity between two entities based on the % that on entity 
    represents over the other.
    '''
    try:
        a1 = row[j] / np.sum(row)
        a2 = row2[i] / np.sum(row2)

    except IndexError:
        a1 = row[0,j] / np.sum(row)
        a2 = row2[i,0] / np.sum(row2)

    return a1, a2



def affinity_maquiavelo(row, row2, grades):
    '''
    Calculates the afifnity between two entities based on the grades of their 
    connected nodes.
    '''
    x_prim = row > 0
    y_prim = row2 > 0
    
    ix = np.sum(grades[x_prim])
    iy = np.sum(grades[y_prim])
    
    res = 1 - abs(ix - iy) / max(ix, iy)

    return res, res


def affinity_networking(row, row2, affinities, x, y):
    '''
    Calculates the affinity between two particles based on a preexisting 
    affinity between their two social groups.
    '''
    x_friends = row > 0
    y_friends = row2 > 0
    
    fx = np.mean(affinities[x_friends, :][:, y])
    fy = np.mean(affinities[y_friends, :][:, x])
    
    #res = 1 - abs(ix - iy) / max(ix, iy)
    return fx, fy

def jaccard_affinity(logits):
    nclasificadores, muestras, clases = logits.shape
    logits = np.argmax(logits, axis = 2)
    resultados = np.zeros((nclasificadores, nclasificadores))

    for i in range(nclasificadores):
        for j in range(nclasificadores):
            comunes = np.sum(np.equal(logits[i, :], logits[j, :]))
            res = comunes / len(logits[i, :])
            resultados[i,j] = res

    return resultados

def connexion2affinity(conex0, af_func=affinity_most_important_ally):
    '''
    Given a connectivity matrix calculates a predefined affinity.
    '''
    import pandas as pd
    if isinstance(conex0, pd.DataFrame):
        conex = np.array(conex0)
    else:
        conex = conex0

    rows, cols = conex.shape
    res = np.zeros(conex.shape)
    grades = np.sum(conex0, axis=0)
    
    for i in range(rows):
        suj = conex[i,:]
        for j in range(cols):
            if i != j:
                suj2 = conex[j,:]
                try:
                    afinidad1, afinidad2 = af_func(suj, suj2) #- 2 * conex[i,j] / cols
                except TypeError:
                     afinidad1, afinidad2 = af_func(suj, suj2, grades)
                res[i, j] = afinidad1
                res[j, i] = afinidad2

    res = np.nan_to_num(res)
    if isinstance(conex0, pd.DataFrame):
        res = pd.DataFrame(res)
        res.index = conex0.index
        res.columns = conex0.columns

    return res


def conex2af_std_comb(conex, alpha=0.5):
    sumatorios = np.sum(conex, axis=1).reshape(conex.shape[0], 1)
    res = np.zeros(conex.shape, dtype=np.float64)
    rows, cols = conex.shape

    for x in range(rows):
        cx = conex[x, :]
        sx = sumatorios[x]
        for y in range(cols):
            cy = conex[y, :]
            aux = np.max(np.minimum(cx, cy)) / sx
            res[x, y] = aux[0]

    res2 = conex / sumatorios

    return res * alpha + (1-alpha) * res2

