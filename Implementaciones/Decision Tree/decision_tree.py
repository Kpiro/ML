import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DecisionTree:
    def __init__(self, max_depth=None):
        """
        Inicializa el árbol de decisión.
        
        :param max_depth: Profundidad máxima del árbol. Si es None, crece indefinidamente.
        """
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Ajusta el árbol a los datos de entrenamiento.
        
        :param X: numpy.ndarray, datos de entrada de tamaño (n_samples, n_features).
        :param y: numpy.ndarray, etiquetas de tamaño (n_samples,).
        """
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        """
        Predice las etiquetas para los datos de entrada.
        
        :param X: numpy.ndarray, datos de entrada de tamaño (n_samples, n_features).
        :return: numpy.ndarray, etiquetas predichas.
        """
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth=0):
        """
        Construye el árbol recursivamente.
        
        :param X: numpy.ndarray, datos de entrada en el nodo actual.
        :param y: numpy.ndarray, etiquetas correspondientes al nodo actual.
        :param depth: Profundidad actual del árbol.
        :return: Diccionario que representa el nodo del árbol.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Criterios de parada
        if n_labels == 1 or n_samples == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return {'label': self._most_common_label(y)}

        # Encontrar la mejor división
        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            return {'label': self._most_common_label(y)}

        # Dividir los datos
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold
        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def _best_split(self, X, y):
        """
        Encuentra la mejor característica y umbral para dividir los datos.
        
        :param X: numpy.ndarray, datos de entrada.
        :param y: numpy.ndarray, etiquetas correspondientes.
        :return: Índice de la característica y umbral óptimos.
        """
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None, None

        # Entropía inicial
        parent_entropy = self._entropy(y)

        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold

                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                # Calcular ganancia de información
                left_entropy = self._entropy(y[left_idx])
                right_entropy = self._entropy(y[right_idx])
                n_left, n_right = len(y[left_idx]), len(y[right_idx])
                child_entropy = (n_left / n_samples) * left_entropy + (n_right / n_samples) * right_entropy
                info_gain = parent_entropy - child_entropy

                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _entropy(self, y):
        """
        Calcula la entropía de un conjunto de etiquetas.
        
        :param y: numpy.ndarray, etiquetas.
        :return: Entropía.
        """
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _most_common_label(self, y):
        """
        Encuentra la etiqueta más común en el conjunto de etiquetas.
        
        :param y: numpy.ndarray, etiquetas.
        :return: Etiqueta más común.
        """
        return np.bincount(y).argmax()

    def _predict_single(self, x, tree):
        """
        Predice la etiqueta para una única muestra.
        
        :param x: numpy.ndarray, muestra de entrada.
        :param tree: Nodo del árbol.
        :return: Etiqueta predicha.
        """
        if 'label' in tree:
            return tree['label']

        feature = tree['feature']
        threshold = tree['threshold']

        if x[feature] <= threshold:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])

# ------------------------------- MAIN ------------------------------------------

# Cargar datos
data = load_iris()
X = data.data
y = data.target

# Convertir el problema a binario (clase 0 vs no clase 0)
y = (y == 0).astype(int)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y ajustar el modelo
tree = DecisionTree(max_depth=3)
tree.fit(X_train, y_train)

# Hacer predicciones
y_pred = tree.predict(X_test)

# Evaluar precisión
print("Precisión del modelo:", accuracy_score(y_test, y_pred))
