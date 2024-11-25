# Análisis de Clasificación con Modelos de Machine Learning

Este proyecto tiene como objetivo aplicar y evaluar diferentes modelos de clasificación para predecir una variable objetivo en un conjunto de datos. 
Se utilizaron técnicas de preprocesamiento, selección de características y validación cruzada para mejorar la precisión del modelo. Los modelos entrenados incluyen K-Nearest Neighbors (KNN), Random Forest y Regresión Logística.

## Propósito del Proyecto

El propósito principal de este proyecto es aplicar algoritmos de machine learning para predecir la clase de una variable de interés a partir de un conjunto de datos con variables numéricas y categóricas. 
A través de este análisis, se evalúa el rendimiento de varios modelos de clasificación y se selecciona el mejor basado en métricas de rendimiento como exactitud, precisión, recall, F1-score, y AUC.

## Técnicas Utilizadas

1. **Carga y Exploración de Datos:**
   - Carga de los datos desde un archivo CSV.
   - Revisión de la estructura y estadísticas básicas.
   - Análisis de la distribución de las variables.
   - Identificación y manejo de valores nulos y outliers.

2. **Preprocesamiento de Datos:**
   - Selección de características relevantes.
   - Conversión de variables categóricas a numéricas (Label Encoding, One-Hot Encoding).
   - Escalado de características para mejorar el rendimiento de ciertos modelos (como KNN).
   - División del dataset en conjuntos de entrenamiento y prueba.

3. **Entrenamiento de Modelos:**
   - Entrenamiento de tres modelos de clasificación: 
     - **K-Nearest Neighbors (KNN)**
     - **Random Forest**
     - **Regresión Logística**
   - Validación cruzada para la selección de los mejores hiperparámetros utilizando `GridSearchCV`.

4. **Evaluación de Modelos:**
   - Cálculo de métricas clave: exactitud, precisión, recall, F1-score.
   - Generación de la matriz de confusión.
   - Curva ROC y cálculo del AUC para evaluar el rendimiento del mejor modelo.

5. **Análisis y Comparación de Resultados:**
   - Comparación del rendimiento de los diferentes modelos.
   - Selección del mejor modelo basado en las métricas.
   - Análisis de las fortalezas y debilidades de cada modelo.

## Cómo Ejecutar el Código

1. **Requisitos previos:**
   - Python 3.16
   - Las siguientes librerías deben estar instaladas:
    pandas
    scikit-learn
    matplotlib
    seaborn

2. **Clonar el Repositorio:**
   - Clona el repositorio en tu máquina local:
   - 
3. **Ejecutar el Código:**
   - Navega a la carpeta del proyecto:
   
   - Ejecuta el archivo Python que contiene el código del análisis:
    
4. **Resultados:**
   - El código generará resultados de evaluación de los modelos (precisión, recall, F1, etc.), matrices de confusión y curvas ROC, que se mostrarán en la terminal o en gráficos.


