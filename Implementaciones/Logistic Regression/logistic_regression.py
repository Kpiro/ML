import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6):
        """
        Inicializa el modelo de Regresión Logística.
        
        :param learning_rate: Tasa de aprendizaje para gradiente descendente
        :param max_iter: Número máximo de iteraciones
        :param tol: Tolerancia para la convergencia
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None  # Pesos del modelo
        self.bias = 0  # Sesgo del modelo

    def sigmoid(self, z):
        """
        Función sigmoide.
        
        :param z: Entrada (puede ser un escalar o un array)
        :return: Valor de la función sigmoide
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Ajusta el modelo a los datos de entrenamiento.
        
        :param X: numpy.ndarray de tamaño (n_samples, n_features)
        :param y: numpy.ndarray de tamaño (n_samples,) con valores 0 o 1
        """
        n_samples, n_features = X.shape
        
        # Inicializar pesos y sesgo
        self.weights = np.zeros(n_features)
        self.bias = 0

        for iteration in range(self.max_iter):
            # Predicción lineal
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Cálculo de gradientes
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Actualización de parámetros
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Verificar convergencia
            if np.linalg.norm(dw) < self.tol and abs(db) < self.tol:
                break

    def predict_proba(self, X):
        """
        Calcula las probabilidades predichas para las muestras de entrada.
        
        :param X: numpy.ndarray de tamaño (n_samples, n_features)
        :return: numpy.ndarray de tamaño (n_samples,) con probabilidades
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Predice etiquetas binarias (0 o 1) para las muestras de entrada.
        
        :param X: numpy.ndarray de tamaño (n_samples, n_features)
        :param threshold: Umbral para clasificar como 1
        :return: numpy.ndarray de tamaño (n_samples,) con etiquetas predichas
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

# ------------------------------- MAIN ---------------------------------------------------

# Generar un conjunto de datos de clasificación binaria
X, y = make_classification(
    n_samples=500,       
    n_features=2,        
    n_informative=2,     
    n_redundant=0,       
    n_repeated=0,        
    n_classes=2,         
    random_state=42      
)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = LogisticRegression(learning_rate=0.1, max_iter=1000)
model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")
