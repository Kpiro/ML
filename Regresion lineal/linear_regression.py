import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, X, y):
        # Hiperparámetros
        learning_rate = 0.01  # Tasa de aprendizaje
        n_iterations = 1000   # Número máximo de iteraciones
        epsilon = 1e-6        # Umbral para la convergencia

        # Agregar una columna de unos a X para el término beta_0 (intercepto)
        X = np.c_[np.ones(X.shape[0]), X]

        # Inicializar parámetros beta (incluye beta_0 y coeficientes de características)
        self.beta = np.zeros(X.shape[1])

        # Número de muestras
        n = X.shape[0]

        # Inicializar variables para el criterio de convergencia
        previous_mse = float('inf')  # Un valor muy alto inicialmente

        # Gradiente descendente
        for iteration in range(n_iterations):
            # Predicción
            y_pred = X.dot(self.beta)

            # Cálculo del error
            error = y_pred - y

            # Cálculo del MSE
            mse = np.mean(error ** 2)

            # Verificar convergencia
            if abs(previous_mse - mse) < epsilon:
                print(f"Convergencia alcanzada en la iteración {iteration}")
                break
            previous_mse = mse  # Actualizar el MSE previo

            # Gradiente respecto a beta
            gradient = (2 / n) * X.T.dot(error)

            # Actualizar parámetros beta
            self.beta -= learning_rate * gradient

            # (Opcional) Imprimir el progreso
            if iteration % 100 == 0:
                print(f"Iteración {iteration}: MSE = {mse:.4f}, Coeficientes = {self.beta}")

        # Resultados finales
        print("\nCoeficientes finales:")
        for i, b in enumerate(self.beta):
            print(f"beta_{i} (parámetro {i}): {b:.4f}")

    def pred(self, new_data):
        new_data = np.array(new_data)
        new_data = np.c_[np.ones(new_data.shape[0]), new_data]
        print('new data', new_data)
        return new_data.dot(self.beta)

# Datos de ejemplo (X: características, y: etiquetas)
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # Dos características
y = np.array([5, 7, 9, 11])  # Valores objetivo

model = LinearRegression(X, y)
print('Predicciones: ', model.pred([
    [2.0, 3.0],
    [1.5, 2.5],
    [3.0, 1.0]
]))

