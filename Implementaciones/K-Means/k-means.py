import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4):
        """
        Inicializa el algoritmo K-Means.
        
        :param n_clusters: número de clusters (k)
        :param max_iter: número máximo de iteraciones
        :param tol: tolerancia para la convergencia
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None  # Centroides de los clusters
        self.labels = None  # Etiquetas asignadas a los puntos

    def fit(self, X):
        """
        Ajusta el modelo a los datos.
        
        :param X: numpy.ndarray de tamaño (n_muestras, n_características)
        """
        n_samples, n_features = X.shape
        
        # Inicializar centroides aleatoriamente a partir de los datos
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        
        for iteration in range(self.max_iter):
            # Asignar cada punto al cluster más cercano
            distances = self._compute_distances(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)
            
            # Calcular nuevos centroides como la media de los puntos del cluster
            new_centroids = np.array([
                X[self.labels == k].mean(axis=0) if len(X[self.labels == k]) > 0 else self.centroids[k]
                for k in range(self.n_clusters)
            ])
            
            # Verificar convergencia
            if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) <= self.tol):
                break
            
            self.centroids = new_centroids

    def predict(self, X):
        """
        Asigna cada punto a su cluster más cercano.
        
        :param X: numpy.ndarray de tamaño (n_muestras, n_características)
        :return: numpy.ndarray de etiquetas asignadas
        """
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

    def _compute_distances(self, X, centroids):
        """
        Calcula las distancias euclidianas entre puntos y centroides.
        
        :param X: numpy.ndarray de tamaño (n_muestras, n_características)
        :param centroids: numpy.ndarray de tamaño (n_clusters, n_características)
        :return: numpy.ndarray de distancias
        """
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

# ---------------------------------- MAIN ----------------------------------------------------

# Generar datos de ejemplo
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Ejecutar K-Means
kmeans = KMeans(n_clusters=4, max_iter=100, tol=1e-4)
kmeans.fit(X)

# Obtener etiquetas de cluster
labels = kmeans.predict(X)

# Visualizar resultados
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='x', s=100, label='Centroides')
plt.title("Clusters encontrados por K-Means")
plt.legend()
plt.show()
