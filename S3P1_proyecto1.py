#!/usr/bin/env python
# coding: utf-8

# ![image info](https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/images/banner_1.png)

# # Proyecto 1 - Predicción de precios de vehículos usados
# 
# En este proyecto podrán poner en práctica sus conocimientos sobre modelos predictivos basados en árboles y ensambles, y sobre la disponibilización de modelos. Para su desasrrollo tengan en cuenta las instrucciones dadas en la "Guía del proyecto 1: Predicción de precios de vehículos usados".
# 
# **Entrega**: La entrega del proyecto deberán realizarla durante la semana 4. Sin embargo, es importante que avancen en la semana 3 en el modelado del problema y en parte del informe, tal y como se les indicó en la guía.
# 
# Para hacer la entrega, deberán adjuntar el informe autocontenido en PDF a la actividad de entrega del proyecto que encontrarán en la semana 4, y subir el archivo de predicciones a la [competencia de Kaggle](https://www.kaggle.com/t/b8be43cf89c540bfaf3831f2c8506614).

# ## Datos para la predicción de precios de vehículos usados
# 
# En este proyecto se usará el conjunto de datos de Car Listings de Kaggle, donde cada observación representa el precio de un automóvil teniendo en cuenta distintas variables como: año, marca, modelo, entre otras. El objetivo es predecir el precio del automóvil. Para más detalles puede visitar el siguiente enlace: [datos](https://www.kaggle.com/jpayne/852k-used-car-listings).

# ## Ejemplo predicción conjunto de test para envío a Kaggle
# 
# En esta sección encontrarán el formato en el que deben guardar los resultados de la predicción para que puedan subirlos a la competencia en Kaggle.

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Importación librerías
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


# In[ ]:


# Carga de datos de archivo .csv
dataTraining = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTrain_carListings.zip')
dataTesting = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTest_carListings.zip', index_col=0)


# In[ ]:


# Visualización datos de entrenamiento
dataTraining.head()


# In[ ]:


# Visualización datos de test
dataTesting.head()


# In[ ]:


# Predicción del conjunto de test - acá se genera un número aleatorio como ejemplo
np.random.seed(42)
y_pred = pd.DataFrame(np.random.rand(dataTesting.shape[0]) * 75000 + 5000, index=dataTesting.index, columns=['Price'])


# In[ ]:


# Guardar predicciones en formato exigido en la competencia de kaggle
y_pred.to_csv('test_submission.csv', index_label='ID')
y_pred.head()


# In[ ]:


y_pred.info


# # Preprocesamiento de datos (10 puntos)

# - Los datos de entrenamiento se dividen en datos de entrenamiento y validación. Si decidieron preprocesar los datos (estandarizar, normalizar, imputar valores, etc), estos son correctamente preprocesados al ajustar sobre los datos de entrenamiento (.fit_transform()) y al transformar los datos del set de validación (.transform()). (10 puntos)

# In[ ]:


# Codificar las variables categóricas de dataTraing
dataTraining["State"], _ = pd.factorize(dataTraining["State"])
dataTraining["Make"], _ = pd.factorize(dataTraining["Make"])
dataTraining["Model"], _ = pd.factorize(dataTraining["Model"])

# Codificar las variables categóricas de dataTesting
dataTesting["State"], _ = pd.factorize(dataTesting["State"])
dataTesting["Make"], _ = pd.factorize(dataTesting["Make"])
dataTesting["Model"], _ = pd.factorize(dataTesting["Model"])

#dataTraining.head()
dataTesting.tail()


# In[ ]:


dataTraining.tail()


# In[ ]:


# Visualizar los valores de las variables en dataTraining

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
dataTraining.plot(kind = "scatter", x="Model", y="Mileage", c="Price", colormap="jet", xlim=(0, 25), ylim=(0, 250))  


# In[ ]:


# Selección de las varibles predictoras
Tr_feature_cols = dataTraining.columns[dataTraining.columns.str.startswith("C")==False].drop("Price")
Tr_feature_cols


# In[ ]:


Te_feature_cols = dataTesting.columns[dataTesting.columns.str.startswith("C")==False]
Te_feature_cols


# In[ ]:


#Describir la variable Price
dataTraining.Price.describe()


# In[ ]:


# Separar las variables predictoras (X) y las varibles de interés (y)
X_train = dataTraining[Tr_feature_cols]
y_train = (dataTraining.Price > 18450).astype(int)


# In[ ]:


X_train


# In[ ]:


X_test = dataTesting[Te_feature_cols]


# In[ ]:


X_test


# In[ ]:


# Implementar el modelo 


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# X_train y y_train corresponde a los datos de las características y las etiquetas para el conjunto de entrenamiento
# y X_test es el conjunto de características para el conjunto de validación.
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

# Crear un objeto StandardScaler para el procesamiento de datos
scaler = StandardScaler()

# Ajustar y transformar X_train
X_train_preprocessed = scaler.fit_transform(X_train)

# Transformar los datos de validación sin considerar ajustes en el modelo de datos
X_val_preprocessed = scaler.transform(X_test)


# # Calibración del modelo (15 puntos)

# - Se calibran los parámetros que se consideren pertinentes del modelo de clasificación seleccionado. (5 puntos)
# - Se justifica el método seleccionado de calibración. (5 puntos)
# - Se analizan los valores calibrados de cada parámetro y se explica cómo afectan el modelo. (5 puntos)

# **RandomForest**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Crear o definir el modelo RandomForest
randomForest_model = RandomForestRegressor(random_state=42)
randomForest_model


# In[ ]:


cross_val_score(randomForest_model, X_train, y_train, cv=10)


# In[ ]:


# Evaluar el desempeño del modelo
rf_performance = cross_val_score(randomForest_model, X_train, y_train, cv=10)
rf_performance_values = pd.Series(rf_performance)
print(rf_performance_values.describe())


# In[ ]:


# Creación de lista de valores para iterar sobre diferentes valores de n_estimators
estimator_range = range(10, 310, 10)

# Definición de lista para almacenar la exactitud (accuracy) promedio para cada valor de n_estimators
accuracy_scores = []

# Uso de un 10-fold cross-validation para cada valor de n_estimators
for estimator in estimator_range:
    randomForest_model = RandomForestRegressor(n_estimators=estimator, random_state=1, n_jobs=-1)
    accuracy_scores.append(cross_val_score(randomForest_model, X_train, y_train, cv=10, scoring='accuracy').mean())
print (accuracy_scores)


# In[ ]:


# Gráfica del desempeño del modelo vs la cantidad de n_estimators
plt.plot(estimator_range, accuracy_scores)
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')


# In[ ]:


# Creación de lista de valores para iterar sobre diferentes valores de max_features
feature_range = range(1, len(feature_cols)+1)

# Definición de lista para almacenar la exactitud (accuracy) promedio para cada valor de max_features
accuracy_scores = []

# Uso de un 10-fold cross-validation para cada valor de max_features
for feature in feature_range:
    clf = RandomForestRegressor(n_estimators=200, max_features=feature, random_state=1, n_jobs=-1)
    accuracy_scores.append(cross_val_score(randomForest_model, X, y, cv=5, scoring='accuracy').mean())


# In[ ]:


# Gráfica del desempeño del modelo vs la cantidad de max_features
plt.plot(feature_range, accuracy_scores)
plt.xlabel('max_features')
plt.ylabel('Accuracy')


# **XGBoots**

# In[ ]:


from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Definir el modelo XGBoost
xgb_model = xgb.XGBRegressor()
print(xgb_model)


# Definir el espacio de búsqueda de hiperparámetros
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5]
}

# Realizar la búsqueda grid
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Obtener los mejores hiperparámetros
best_params = grid_search.best_params_
print("Mejores hiperparámetros encontrados:", best_params)


# # Entrenamiento del modelo (15 puntos)

# - Se entrena el modelo de clasificación escogido con los datos del set de entrenamiento preprocesados y los parámetros óptimos. (5 puntos)
# - Se presenta el desempeño del modelo en los datos de validación con al menos una métrica de desempeño. (5 puntos)
# - Se justifica la selección del modelo correctamente. (5 puntos)
# 

# **RandomForest**

# In[ ]:


#Desarrollo del entregamiento del modelo Random Forest

# Entrenar el modelo Random Forest utilizando los datos de entrenamiento
randomForest_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de datos de prueba
predictions = randomForest_model.predict(X_test)

rf_predictions_df = pd.DataFrame(predictions, index=X_test.index, columns=['Price'])

rf_predictions_df.to_csv('rf_test_submission.csv', index_label='ID')


# **XGBoots**

# In[ ]:


# Entrenar el modelo utilizando los datos de entrenamiento
xgb_model = xgb.XGBRegressor(**best_params)
xgb_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de datos de prueba
predictions = xgb_model.predict(X_test)

xgb_predictions_df = pd.DataFrame(predictions, index=X_test.index, columns=['Price'])

xgb_predictions_df.to_csv('xgb_test_submission.csv', index_label='ID')


# # Disponibilizar modelo con Flask (30 puntos)

# - Para esta sección del notebook instale las siguientes librerías !pip install flask y !pip install flask_restplus.    Utilizamos pip install flask-restful.
# - Se disponibiliza el modelo en una API alojada en un servicio en la nube. (20 puntos)
# - Se hacen las predicciones sobre el valor del automóvil en al menos dos observaciones del set de validación. (10 puntos)
# 

# In[ ]:


# Importación librerías
from flask import Flask
from flask_restful import Api, Resource, fields


# In[ ]:


#Desarrollo de la disponibilización del modelo

 Definición aplicación Flask
app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='API predicción precios vehículos',
    description='API predicción precios vehículos')

ns = api.namespace('predict', 
     description='Predicción precios vehículos')

# Definición argumentos o parámetros de la API
parser = api.parser()
parser.add_argument(
    'URL', 
    type=str, 
    required=True, 
    help='URL to be analyzed', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})


# In[ ]:


# Definición de la clase para disponibilización
@ns.route('/')
class vehiculosApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_proba(args['URL'])
        }, 200


# In[ ]:


# Ejecución de la aplicación que disponibiliza el modelo de manera local en el puerto 5000
app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)


# # Conclusiones (10 puntos)

# 

# In[ ]:





# In[ ]:




