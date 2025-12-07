#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy  as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("airbnb-listings-extract.csv", sep=";")

print("Número de filas y columnas:", df.shape)
df.head(5).T


# In[3]:


df.info()


# In[4]:


df.describe().T


# In[5]:


df.dtypes


# In[6]:


df.isnull().sum().sort_values(ascending=False).head(30)


# In[7]:


(df.isnull().mean() * 100).sort_values(ascending=False).head(30)


# In[8]:


target = "Price"


# In[9]:


plt.figure(figsize=(8,4))
plt.hist(df[target].dropna(), bins=50)
plt.xlabel(target)
plt.ylabel("Frecuencia")
plt.title("Distribución de la variable objetivo (Price)")
plt.show()


# In[10]:


df.columns


# In[11]:


cols_to_drop_manual = [
    # IDs y URLs
    "ID", "Listing Url", "Scrape ID",
    "Thumbnail Url", "Medium Url", "Picture Url", "XL Picture Url",
    "Host ID", "Host URL", "Host Thumbnail Url", "Host Picture Url",
    "Geolocation",
    # Textos largos / descripciones
    "Name", "Summary", "Space", "Description", "Experiences Offered",
    "Neighborhood Overview", "Notes", "Transit", "Access", "Interaction",
    "House Rules", "Host Name", "Host About", "Host Location", "Street",
    "Amenities", "Features",
    # Otros campos administrativos poco útiles
    "License", "Jurisdiction Names", "Cancellation Policy",
    # Precios redundantes
    "Weekly Price", "Monthly Price"
]

df = df.drop(columns=cols_to_drop_manual, errors="ignore")
df.shape


# In[12]:


missing_pct = df.isnull().mean().sort_values(ascending=False)
missing_pct.head(20)


# In[13]:


cols_many_missing = missing_pct[missing_pct > 0.5].index.tolist()
cols_many_missing


# In[14]:


df = df.drop(columns=cols_many_missing)
df.shape


# In[15]:


missing_pct = df.isnull().mean().sort_values(ascending=False)
missing_pct.head(20)


# In[16]:


df = df.dropna(subset=["Price"])
df.shape, df["Price"].isnull().sum()


# In[17]:


target = "Price"
numeric_features = [
    "Host Response Rate", "Host Listings Count", "Host Total Listings Count",
    "Latitude", "Longitude", "Accommodates", "Bathrooms", "Bedrooms", "Beds",
    "Cleaning Fee", "Guests Included", "Extra People",
    "Minimum Nights", "Maximum Nights",
    "Availability 30", "Availability 60", "Availability 90", "Availability 365",
    "Number of Reviews",
    "Review Scores Rating", "Review Scores Accuracy", "Review Scores Cleanliness",
    "Review Scores Checkin", "Review Scores Communication",
    "Review Scores Location", "Review Scores Value",
    "Calculated host listings count", "Reviews per Month"
]
print("Número de variables numéricas:", len(numeric_features))
numeric_features


# In[18]:


corr_con_price = df[numeric_features + [target]].corr()[target].sort_values(ascending=False)
print("Correlación de cada variable numérica con el precio:")
corr_con_price


# In[19]:


df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())
df[numeric_features].isnull().sum().sort_values(ascending=False).head()


# In[20]:


from sklearn.model_selection import train_test_split
X = df[numeric_features].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=2
)
print("X_train:", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train:", y_train.shape)
print("y_test: ", y_test.shape)


# In[21]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# In[22]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(Lasso(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train_scaled, y_train)
print("Mejor alpha:", grid.best_params_)
print("Mejor score:", grid.best_score_)


# In[23]:


from sklearn.metrics import mean_squared_error
alpha_optimo = grid.best_params_['alpha']
print("Alpha óptimo:", alpha_optimo)
lasso = Lasso(alpha=alpha_optimo).fit(X_train_scaled, y_train)
y_train_lasso = lasso.predict(X_train_scaled)
y_test_lasso  = lasso.predict(X_test_scaled)
mse_train_lasso = mean_squared_error(y_train, y_train_lasso)
mse_test_lasso  = mean_squared_error(y_test, y_test_lasso)
print('MSE Modelo Lasso (train): %0.3g' % mse_train_lasso)
print('MSE Modelo Lasso (test) : %0.3g' % mse_test_lasso)
print('RMSE Modelo Lasso (train): %0.3g' % np.sqrt(mse_train_lasso))
print('RMSE Modelo Lasso (test) : %0.3g' % np.sqrt(mse_test_lasso))
w = lasso.coef_
for f, wi in zip(numeric_features, w):
    print(f, wi)


# In[24]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    random_state=2,
    n_jobs=-1
)

scores = cross_val_score(
    rf,
    X_train, y_train,
    cv=5,
    scoring='neg_mean_squared_error'
)

print("MSE CV (folds):", -scores)
print("MSE CV medio  :", -scores.mean())
print("Std CV        :", scores.std())


# In[25]:


rf.fit(X_train, y_train)
y_train_rf = rf.predict(X_train)
y_test_rf  = rf.predict(X_test)
mse_train_rf = mean_squared_error(y_train, y_train_rf)
mse_test_rf  = mean_squared_error(y_test,  y_test_rf)
print('MSE Random Forest (train): %0.3g' % mse_train_rf)
print('MSE Random Forest (test) : %0.3g' % mse_test_rf)
print('RMSE Random Forest (train): %0.3g' % np.sqrt(mse_train_rf))
print('RMSE Random Forest (test) : %0.3g' % np.sqrt(mse_test_rf))


# In[26]:


# Celda 1 – Importación de librerías y configuración inicial

# Celda 2 – Carga del dataset y vista previa, mostrando filas y columnas

# Celda 3 – Información general del DataFrame 

# Celda 4 – Estadísticas descriptivas (df.describe().T)
# Obtengo estadísticas básicas como  media, desviación, percentiles,etc.

# Celda 5 – Tipos de datos (df.dtypes)En esta celda reviso el tipo de cada columna. 
# Esto sirve para saber qué columnas son numéricas y cuáles son texto,äutil para el postprocesamiento

# Celda 6 – Conteo de valores nulos por columna (df.isnull().sum())

# Celda 7 – Porcentaje de nulos por columna

# Celda 8 – Definición de la variable objetivo y visualización de su distribución. en este caso "price"

# Celda 9 – Visualización de la distribución del precio

# Celda 10 – Listado de columnas del DataFrame (df.columns)
# Muestro todas las columnas disponibles y revisando los nombres puedo decidir cuales eliminar

# Celda 11 – Eliminación manual de columnas que no necesito

# Celda 12 – Cálculo del porcentaje de nulos tras la primera limpieza y detecciñon de columnas con muchos valores cero.

# Celda 13 – Detección de columnas con demasiados nulos, aplicando regla de 50%

# Celda 14 – Eliminación de estas columnas con más del 50% de nulos

# Celda 15 – Nueva revisión de nulos tras eliminar columnas

# Celda 16 – Eliminación de filas sin precio en columna "price"

# Celda 17 – definición de las variables numéricas que usaré para el modelo

# Celda 18 – Correlación entre variables numéricas y el precio para interpretar el problema a solucionar

# Celda 19 – Imputación de valores nulos en variables numéricas. con esto compruebo que za no haz valores nulos

# Celda 20 – Separación en entrenamiento y test

# Celda 21 – Aplico StandardScaler sobre X_train y X_test para obtener X_train_scaled
# y X_test_scaled. 

# Celda 22 – Entrenamiento del modelo Lasso con GridSearchCV, para encontrar el mejor valor de alpha para el modelo Lasso.

# Celda 23 – Evaluación detallada del modelo Lasso,entrenándolo con el mejor alpha y evaluando su rendimiento en train y test
# y analizo los coeficientes para interpretar la contribución de cada variable al precio.

# Celda 24 – Definición y validación cruzada del modelo Random Forest,
# evaluándolo mediante validación cruzada para obtener una estimación fiable de su rendimiento medio y compararlo con Lasso.

# Celda 25 – Entrenamiento final y evaluación del Random Forest.
# En esta celda entreno y evalúo el Random Forest definitivo y comparo su desempeño con el del modelo Lasso, verificando que Random Forest logra menor error en test.


# In[27]:


# CONCLUSIÓN
# En esta práctica he limpiado el dataset y preparado las variables para poder aplicar modelos de regresión.
# Probé Lasso y Random Forest y, aunque ambos funcionan, prefiero los resultados obtenidos con Random Forest.
# Lasso resultó útil para identificar qué variables tienen mayor influencia sobre el precio.
# En conjunto, el flujo completo de análisis me permitió construir y comparar dos modelos distintos y entender mejor el comportamiento del dataset.

