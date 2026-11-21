import numpy as np

from svm import (
    svm_cu_dual_fitness,
    verifica_constrangeri,
    repara_alfa,
    calculeaza_w,
    calculeaza_b,
    prezice_svm
)

C = 1.0

# Test 1 - Verificarea constrangerilor

alfa_test = np.array([0.2, 0.5, 0.3, 0.4])
y_mic = np.array([-1, -1, 1, 1])

print("Test constrangeri:",
      verifica_constrangeri(alfa_test, y_mic, C))


# Test 2 - Repararea multiplicatorilor

alfa_invalid = np.array([0.2, 0.5, 0.3, 0.8])

alfa_reparat = repara_alfa(alfa_invalid, y_mic, C)

print("Test reparare:",
      verifica_constrangeri(alfa_reparat, y_mic, C))


# Test 3 - Calcularea parametrilor SVM

X_test_mic = np.array([
    [-1.0, 0.0],
    [1.0, 0.0]
])

y_test_mic = np.array([-1, 1])
alfa_mic = np.array([0.5, 0.5])

w = calculeaza_w(alfa_mic, X_test_mic, y_test_mic)
b = calculeaza_b(alfa_mic, X_test_mic, y_test_mic, w, C)

predictii = prezice_svm(X_test_mic, w, b)

print("w:", w)
print("b:", b)
print("Predictii:", predictii)

assert np.array_equal(predictii, y_test_mic)
print("Testele simple au trecut!")