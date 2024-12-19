## Métricas de distancia que se pueden utilizar con el algoritmo kNN

1. Distancia Euclideana

$$ d(a,b) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^{2}} $$

2. Distancia de Manhattan

$$ d(a,b) = \sum_{i=1}^{n} |a_i - b_i| $$


3. Coeficiente de Jaccard

$$ d(A,B) = (|A \cap B|) \div (|A| + |B| - |A \cap B|) $$

4. Distancia de Minkowski 

$$ d(a,b) = { (\sum_{i=1}^{n} {|a_i - b_i|}^{p}) }^{1/p} $$

n: tamaño de los vectores a y b

p: parámetro de Minkowski (para p=1 es igual a la distancia de Manhattan y para p=2 es igual a la distancia Euclideana)

## Algoritmo de kNN

1. Calcular la distancia del punto dado hacia todos los demás puntos.
2. Quedarse con los k puntos más cercanos.
3. En términos de clasificación, la respuesta es la moda de los grupos de clasificación. Cuando dos o más grupos son igualmente frecuentes, se decide aumentar el k hasta obtener un único grupo de clasificación. 