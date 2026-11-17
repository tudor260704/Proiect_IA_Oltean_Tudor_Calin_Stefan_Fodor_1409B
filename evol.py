from dataclasses import dataclass
import numpy as np


@dataclass
class EvolParams:
    no_chrom:int
    crossover_prob:float
    mutation_prob:float
    no_generations:int

@dataclass
class Chromosome:
    values:np.ndarray
    fitness:float

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
           a_new[k] = a_new[k] - s 
        else:
           a_new[k] = 0
        
        s = np.dot(a_new,y)
    return a_new


#notam cu R partea din functia de fitness care depinde de X si y care este constanta pentru toti cromozomii
def precompute_R(X:np.ndarray,y:np.ndarray) -> np.ndarray:
    K = X @ X.T #obtinem xiT*xj in matricea K
    Y = np.outer(y,y) # obtinem yi*yj in matricea Y
    return Y*K

def fitness(a:np.ndarray,R:np.ndarray):
    s = a @ R @ a     
    return -np.sum(a) + 0.5*s
    
def init(params:EvolParams,y:np.ndarray,R:np.ndarray,l:int,C:int):
    pop = []
    for _ in range(params.no_chrom):
        chrom_values = generate_individual(l,C)
        chrom_values = adjust_individual(chrom_values,y)
        chrom_fitness = fitness(chrom_values,R)
        pop.append(Chromosome(chrom_values,chrom_fitness))
    return pop

def evolutionary_algorithm(
    X,y,C:int,params:EvolParams
):
    R = precompute_R(X,y)
    for _ in range(params.no_generations):
