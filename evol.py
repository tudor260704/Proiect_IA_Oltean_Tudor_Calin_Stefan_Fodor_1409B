from dataclasses import dataclass
from typing import Any
import numpy as np

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)


@dataclass
class EvolParams:
    no_chrom:int
    crossover_prob:float
    mutation_prob:float
    no_generations:int

@dataclass(init=False)
class Chromosome:
    values:np.ndarray
    fitness:float
    id = 0
    next_id = 0
    def __init__(self, values:np.ndarray) -> None:
        self.values = values
        self.fitness = 0
        self.id = self.next_id
        Chromosome.next_id += 1
    

def generate_individual(l:int,C:int):
    return rng.uniform(0,C,l)


def delta_plus(yi:int):
    if yi == 1:
       return 1
    return 0

def delta_minus(yi:int):
    if yi == -1:
       return 1
    return 0  

def compute_splus(a:np.ndarray,y:np.ndarray):
    ay = a*y
    vectorized_delta = np.vectorize(delta_plus)
    delta_y = vectorized_delta(y)
    return np.dot(ay,delta_y)

def compute_sminus(a:np.ndarray,y:np.ndarray):
    ay = a*y
    vectorized_delta = np.vectorize(delta_minus)
    delta_y = vectorized_delta(y)
    return np.dot(ay,delta_y)

def adjust_individual(a:np.ndarray,y:np.ndarray,eps=1e-8):
    a_new = a.copy()
    s = np.dot(a,y)
    l = len(a)
    
    while abs(s) > eps:
        s_plus = compute_splus(a_new,y)
        s_minus = compute_sminus(a_new,y)
        
        if s_plus > abs(s_minus):
           k = np.random.randint(0,l) 
           while y[k] != 1:
             k = np.random.randint(0,l) 
        else:
            k = np.random.randint(0,l) 
            while y[k] != -1:
              k = np.random.randint(0,l)
        
        if a_new[k] > abs(s):
           a_new[k] = a_new[k] - abs(s) 
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
    
def init_pop(params:EvolParams,y:np.ndarray,l:int,C:int):
    pop = []
    for _ in range(params.no_chrom):
        chrom_values = generate_individual(l,C)
        chrom_values = adjust_individual(chrom_values,y)
        pop.append(Chromosome(chrom_values))
    return pop

def calculate_pop_fitness(pop:list[Chromosome],R:np.ndarray):
    for chromosome in pop:
        chromosome.fitness = fitness(chromosome.values,R)
    return pop

def most_adapted_individual(pop:list[Chromosome]):
    sorted_pop = sorted(pop,key=lambda c : c.fitness) #se urmareste obtinerea valorii minime
    return sorted_pop[0]

def selection_competition(pop:list[Chromosome],k=2):
    len_pop = len(pop)
    parents = []
    for _ in range(2*len_pop):
        competition = []
        competition_ids = [] #adaug indecsii cromozomilor intr-o lista pentru a nu avea coliziuni
        
        for _ in range(k):
            random_idx = rng.integers(0,len_pop)
            
            while random_idx in competition_ids: 
                random_idx = rng.integers(0,len_pop)
            
            chrom = pop[random_idx]
            competition_ids.append(random_idx)
            
            competition.append(chrom)
            
        parents.append(most_adapted_individual(competition))
    return parents

def crossover_arithmetic(p1:Chromosome,p2:Chromosome,crossover_prob:float)->tuple[Chromosome,Chromosome]:
    if crossover_prob < rng.uniform():
       if rng.uniform() < 0.5:
          return (p1,p2)
       else:
           return (p2,p1)
    
    a = rng.uniform()
    b = 1-a
    
    c1_values = np.add(a*p1.values,b*p2.values)
    c2_values = np.add(b*p1.values,a*p2.values)
    
    return (Chromosome(c1_values),Chromosome(c2_values))  

def mutation_reset(c:Chromosome,mutation_prob:float,C:int):
    chrom_len = len(c.values)
    for i in range(chrom_len):
        if mutation_prob > rng.random():
           c.values[i] = rng.uniform(0,C)
    return c

def evolutionary_algorithm(
    X,y,C:int,params:EvolParams
):
    l = len(y)
    R = precompute_R(X,y)
    pop = init_pop(params,y,l,C)
    pop = calculate_pop_fitness(pop,R)
    
    for i in range(params.no_generations):
        print(f"Generation {i+1}")
        
        elite = most_adapted_individual(pop)
        pop_new = [elite]
        parents = selection_competition(pop)
        
        print(f"Most adapted individual {elite.id} -> Fitness {elite.fitness}")
        
        for _ in range(params.no_chrom):
            p1 = parents.pop(0)
            p2 = parents.pop(0)
            
            c1,c2 = crossover_arithmetic(p1,p2,params.crossover_prob)
            c1 = mutation_reset(c1,params.mutation_prob,C)
            c1.values = adjust_individual(c1.values,y)
            
            c2 = mutation_reset(c2,params.mutation_prob,C)
            c2.values = adjust_individual(c2.values,y)
            
            pop_new.append(c1)
            pop_new.append(c2)
        pop_new = calculate_pop_fitness(pop_new,R)
        pop = pop_new
        
    solution = most_adapted_individual(pop)
    return solution

if __name__ == '__main__':
    X_test = np.array([
        [1.0, 1.0], [1.5, 1.0], [1.0, 1.5],   # Clasa -1 (stânga-jos)
        [4.0, 4.0], [4.5, 4.0], [4.0, 4.5]    # Clasa +1 (dreapta-sus)
    ])

    y_test = np.array([-1, -1, -1, 1, 1, 1])
    
    C = 100
    
    problem_params = EvolParams(
        no_chrom=50,
        crossover_prob=0.9,
        mutation_prob=0.1,
        no_generations=100
    )
    
    solution = evolutionary_algorithm(X_test,y_test,C,problem_params)
    a = solution.values
    is_in_interval = np.all((a >= 0) & (a <= C)) 
    print(30*'-')
    print(f"SOLUTION Individual {solution.id} -> Fitness {solution.fitness}")
    print(f"All elements in interval [0,{C}] -> {is_in_interval}")
    print(f"dot(a,y)={np.dot(a,y_test)}")
    print(30*"-")