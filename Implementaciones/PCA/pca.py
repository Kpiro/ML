import numpy as np
from sklearn.datasets import load_iris

class PCA:
    def __init__(self, n_components):
        """
        Inicializa el modelo PCA.
        
        :param n_components: Número de componentes principales a mantener.
        """
        self.n_components = n_components
        self.components = None  # Componentes principales (autovectores)
        self.mean = None  # Media de los datos

    def fit(self, X):
        """
        Ajusta el modelo a los datos de entrada.
        
        :param X: numpy.ndarray de tamaño (n_samples, n_features).
        """
        # Centrar los datos (restar la media)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Calcular la matriz de covarianza
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Obtener autovalores y autovectores
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Ordenar los autovalores y autovectores en orden descendente
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Seleccionar los primeros n_components autovectores
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        """
        Transforma los datos al espacio reducido.
        
        :param X: numpy.ndarray de tamaño (n_samples, n_features).
        :return: numpy.ndarray de tamaño (n_samples, n_components).
        """
        # Centrar los datos
        X_centered = X - self.mean
        # Proyectar los datos en los componentes principales
        return np.dot(X_centered, self.components)

# ---------------------------------- MAIN ------------------------------------------------

# Cargar un conjunto de datos de ejemplo
data = load_iris()
X = data.data  # Características
y = data.target  # Etiquetas

# Crear y ajustar el modelo PCA
pca = PCA(n_components=2)
pca.fit(X)

# Transformar los datos
X_reduced = pca.transform(X)

# Mostrar los datos transformados
print("Datos reducidos (primeras 5 filas):")
print(X_reduced[:5])
