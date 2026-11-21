import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def pregateste_date_iris():
    iris = load_iris()

    # Selectam lungimea si latimea petalei
    X = iris.data[:, [2, 3]]
    y = iris.target

    # Pastram Versicolor si Virginica
    masca = np.isin(y, [1, 2])
    X = X[masca]
    y = y[masca]

    # Transformam clasele in -1 si +1
    y = np.where(y == 1, -1, 1)

    # Impartim datele
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=26,
        stratify=y
    )

    # Standardizam datele
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = pregateste_date_iris()

    print("Date de antrenare:", X_train.shape)
    print("Date de test:", X_test.shape)
    print("Clase antrenare:", np.unique(y_train, return_counts=True))


if __name__ == "__main__":
    main()