import numpy as np

class NaiveBayes:
    def __init__(self):
        self.classes = None        # Clases únicas
        self.feature_probs = {}    # Probabilidades condicionales P(X|Y)
        self.class_priors = {}     # Probabilidades a priori P(Y)
    
    def fit(self, X, y):
        """
        Ajusta el modelo a los datos de entrenamiento.
        
        :param X: numpy.ndarray de tamaño (n_muestras, n_características), valores categóricos
        :param y: numpy.ndarray de tamaño (n_muestras,), etiquetas
        """
        self.classes = np.unique(y)
        
        # Calcular las probabilidades a priori P(Y)
        for cls in self.classes:
            self.class_priors[cls] = np.sum(y == cls) / len(y)
        
        # Calcular las probabilidades condicionales P(X|Y)
        for cls in self.classes:
            X_c = X[y == cls]  # Filtrar las muestras de la clase cls
            self.feature_probs[cls] = {}
            
            for feature_idx in range(X.shape[1]):
                values, counts = np.unique(X_c[:, feature_idx], return_counts=True)
                total = np.sum(counts)
                probs = {val: count / total for val, count in zip(values, counts)}
                
                # Manejo de valores nunca vistos (Laplace smoothing)
                self.feature_probs[cls][feature_idx] = probs
    
    def _compute_posterior(self, x):
        """
        Calcula la probabilidad posterior P(Y|X) para cada clase dada una muestra.
        
        :param x: numpy.ndarray, una muestra con valores categóricos
        :return: clase con mayor probabilidad
        """
        posteriors = {}
        
        for cls in self.classes:
            # Iniciar con el logaritmo del prior log(P(Y))
            posterior = np.log(self.class_priors[cls])
            
            for feature_idx, value in enumerate(x):
                # Extraer P(X|Y) para la característica actual
                probs = self.feature_probs[cls][feature_idx]
                posterior += np.log(probs.get(value, 1e-6))  # Suavizado para valores no vistos
            
            posteriors[cls] = posterior
        
        # Retornar la clase con mayor probabilidad
        return max(posteriors, key=posteriors.get)
    
    def predict(self, X):
        """
        Predice las etiquetas para los datos de entrada.
        
        :param X: numpy.ndarray de tamaño (n_muestras, n_características)
        :return: numpy.ndarray de predicciones
        """
        return np.array([self._compute_posterior(x) for x in X])