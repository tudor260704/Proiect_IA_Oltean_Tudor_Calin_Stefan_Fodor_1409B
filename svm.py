import numpy as np
def svm_cu_dual_fitness(alfa,X,y):
    termenul_liniar=np.sum(alfa)
    K=X @ X.transpose()
    alfa_y=alfa * y
    termenul_patratic=0.5 * (alfa_y @ K @ alfa_y)
    return float(termenul_liniar - termenul_patratic)
def verifica_constrangeri(alfa, y, C):
    if len(alfa)!=len(y):
        return False
    cnt=0
    suma=0
    for alfa_i in alfa:
        if 0<=alfa_i<=C:
            cnt+=1
    for alfa_i,y_i in zip(alfa,y):
        suma+=alfa_i*y_i

    if cnt==len(alfa) and abs(suma)<1e-8:
        return True
    else:
        return False
def repara_alfa(alfa,y,C):
    y=np.asarray(y)

    if not np.all(np.isin(y, [-1,1])):
        raise ValueError("clasele trebuie sa fie -1 sau 1")
    if C <=0:
        raise ValueError("Limita superioara a constrangerii multiplicatorii langrange trebuie sa fie un numar pozitiv")
    alfa=np.array(alfa, dtype=float, copy=True)
    if len(alfa)!=len(y):
        raise ValueError("alfa si y trebuie sa aibe aceiasi lungime!")
    alfa=np.clip(alfa,0,C)
    suma=0
    for alfa_i,y_i in zip(alfa,y):
        suma+=alfa_i*y_i
    if abs(suma) < 1e-8:
        return alfa
    if suma > 0:
        clasa = 1
    else:
        clasa = -1
    pozitii = np.where((y == clasa) & (alfa > 0))[0]
    pozitii = np.random.permutation(pozitii)
    diferenta=abs(suma)
    for i in pozitii:
        reducere=min(alfa[i], diferenta)

        alfa[i] -= reducere
        diferenta -= reducere

        if diferenta < 1e-8:
            break
    return alfa
def calculeaza_w(alfa,X,y):
    alfa_y=alfa * y
    w=X.transpose() @ alfa_y
    return w
def calculeaza_b(alfa,X,y,w,C):
    eps=1e-6

    masca=(alfa>eps) & (alfa< C -eps)
    if not np.any(masca):
        raise ValueError("Nu exista vectori suport liberi")
    valori_b=y[masca]-X[masca] @ w
    b=np.mean(valori_b)
    return float(b)
def prezice_svm(X,w,b):
    scoruri =X @ w +b
    predictii=np.where(scoruri >=0 ,1 ,-1 )
    return predictii