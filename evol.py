from dataclasses import dataclass
import numpy as np


@dataclass
class EvolParams:
    no_chrom:int
    crossover_prob:float
    mutation_prob:float
    no_generations:int

def generate_individual(l:int,C:int):
    return np.random.uniform(0,C,l)


def delta_plus(yi:int):
    if yi == 1:
       return 1
    return 0

def delta_minus(yi:int):
    if yi == -1:
       return 1
    return 0  

def compute_splus(a:np.ndarray,y:np.ndarray):
    ay = np.dot(a,y)
    vectorized_delta = np.vectorize(delta_plus)
    delta_y = vectorized_delta(y)
    return np.dot(ay,delta_y)

def compute_sminus(a:np.ndarray,y:np.ndarray):
    ay = np.dot(a,y)
    vectorized_delta = np.vectorize(delta_minus)
    delta_y = vectorized_delta(y)
    return np.dot(ay,delta_y)

def adjust_individual(a:np.ndarray,y:np.ndarray,eps=1e-8):
    a_new = a.copy()
    s = np.dot(a,y)
    l = len(a)
    
    while s > eps:
        s_plus = compute_splus(a,y)
        s_minus = compute_sminus(a,y)
        
        if s_plus > s_minus:
           k = np.random.randint(0,l) 
           while y[k] != 1:
            k = np.random.randint(0,l) 
        else:
            k = np.random.randint(0,l) 
            while y[k] != -1:
              k = np.random.randint(0,l)
        
        if a_new[k] > s:
           a_new[k] = a_new[k] 
        else:
           a_new[k] = 0
        
        s = np.dot(a_new,y)
    return a_new

def evolutionary(
    X,y,C:int,params:EvolParams
):
    pass