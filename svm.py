from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
iris=load_iris()
print(iris.data.shape)
print(iris.feature_names)
X=iris.data[:,[2,3]]
print(X.shape)
print(X[:5])
y=iris.target
print(iris.target_names)
print(y[:10])
masca=np.isin(y, [1,2])
X=X[masca]
y=y[masca]
print("Dimensiunea X duupa mascare:", X.shape)
print("Dimensiunea y dupa mascare:", y.shape)
y=np.where(y==1,-1,1)
print(np.unique(y, return_counts=True))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26,stratify=y)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Media:", X_train.mean(axis=0))
print("Deviatia standard:", X_train.std(axis=0))
C = 1.0
alpha = np.full(len(y_train), 0.1)

print(alpha.shape)
print(alpha[:5])
fitness=svm_cu_dual_fitness(alpha, X_train, y_train)
print("Fitness SVM:", fitness)

alfa_test = np.array([0.2, 0.5, 0.3, 0.4])
y_test_mic = np.array([-1, -1, 1, 1])

print(verifica_constrangeri(alfa_test, y_test_mic, C))

alfa_invalid = np.array([0.2, 0.5, 0.3, 0.8])

alfa_reparat = repara_alfa(alfa_invalid, y_test_mic, C)

print("Inainte:", alfa_invalid)
print("Dupa:", alfa_reparat)

print("Valid:",
      verifica_constrangeri(alfa_reparat, y_test_mic, C))

alfa_invalid2 = np.array([0.2, 0.8, 0.3, 0.3])

alfa_reparat2 = repara_alfa(alfa_invalid2, y_test_mic, C)

print("Valid dupa reparare:",
      verifica_constrangeri(alfa_reparat2, y_test_mic, C))
alfa_80 = np.random.uniform(0, C, len(y_train))

alfa_80_reparat = repara_alfa(alfa_80, y_train, C)

print("Valid inainte:", verifica_constrangeri(alfa_80, y_train, C))
print("Valid dupa:", verifica_constrangeri(alfa_80_reparat, y_train, C))
