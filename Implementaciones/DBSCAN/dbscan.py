import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        """
        Inicializa el algoritmo DBSCAN.
        
        :param eps: radio máximo para considerar vecinos
        :param min_samples: número mínimo de puntos para formar un núcleo
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None  # Etiquetas de cluster para cada punto

    def fit(self, X):
        """
        Ajusta el modelo a los datos.
        
        :param X: numpy.ndarray de tamaño (n_muestras, n_características)
        """
        n_samples = X.shape[0]
        self.labels = -1 * np.ones(n_samples)  # Inicializar con -1 (ruido)
        cluster_id = 0
        
        for i in range(n_samples):
            if self.labels[i] != -1:  # Ya procesado
                continue
            
            neighbors = self._region_query(X, i)
            
            if len(neighbors) < self.min_samples:  # Ruido
                self.labels[i] = -1
            else:  # Crear un nuevo cluster
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1
    
    def _region_query(self, X, idx):
        """
        Encuentra vecinos dentro de un radio eps para un punto dado.
        
        :param X: numpy.ndarray de tamaño (n_muestras, n_características)
        :param idx: índice del punto para buscar vecinos
        :return: lista de índices de los vecinos
        """
        distances = np.linalg.norm(X - X[idx], axis=1)
        return np.where(distances <= self.eps)[0]
    
    def _expand_cluster(self, X, idx, neighbors, cluster_id):
        """
        Expande un cluster agregando vecinos densos.
        
        :param X: numpy.ndarray de tamaño (n_muestras, n_características)
        :param idx: índice del punto inicial
        :param neighbors: lista de índices de vecinos iniciales
        :param cluster_id: identificador del cluster actual
        """
        self.labels[idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            if self.labels[neighbor_idx] == -1:  # Era ruido, ahora lo incluimos
                self.labels[neighbor_idx] = cluster_id
            
            if self.labels[neighbor_idx] == -1 or self.labels[neighbor_idx] == -1 * np.ones(1):
                self.labels[neighbor_idx] = cluster_id
            
            # Si no está asignado a ningún cluster
            if self.labels[neighbor_idx] == -1:  # No procesado aún
                self.labels[neighbor_idx] = cluster_id
                neighbor_neighbors = self._region_query(X, neighbor_idx)
                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors = np.append(neighbors, neighbor_neighbors)
            
            i += 1

    def predict(self, X):
        """
        Devuelve las etiquetas de cluster asignadas.
        
        :param X: numpy.ndarray de tamaño (n_muestras, n_características)
        :return: numpy.ndarray de etiquetas
        """
        if self.labels is None:
            raise ValueError("El modelo debe ser ajustado primero con `fit`.")
        return self.labels
    
# ---------------------------------------- MAIN --------------------------------------------

# Crear datos de ejemplo
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# Ejecutar DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan.fit(X)

# Obtener etiquetas de cluster
labels = dbscan.predict(X)

# Visualizar resultados
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.title("Clusters encontrados por DBSCAN")
plt.show()
