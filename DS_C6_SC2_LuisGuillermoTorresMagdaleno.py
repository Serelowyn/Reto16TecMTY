# ----------------------- Importaciones

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib import cm as cm
# ----------------------- Fin de las importaciones

# ----------------------- 2. Carga los datos del archivo.

empleados = pd.read_csv("Train.csv")

# a continuacion se realizaran algunas pruebas para poder conocer la forma del df general, y ademas de tipos de datos y ver si se necesitan cambios adicionales

empleados.shape
empleados.dtypes
empleados.info()
empleados.columns
"""aqui se identificaron mayusculas en todos los nombres de columnas, entonces lo limpio para seguir"""

empleados.columns = empleados.columns.str.lower()
empleados.columns

"""esto me sirve para ver la cantidad de celdas con faltantes, en caso de que se requiera despues"""
empleados.isna().sum()

empleados.sample(10)

# ----------------------- 3. Transforma los datos según las características de las variables:

"""para el caso de (employee_id), se sabe que es un elemento primario, que sirve de identificador y es unico por empleado lo cual no nos sirve para los grupos. los elimino"""

empleados = empleados.drop(columns=["employee_id"])
empleados.sample(10)

"""las siguientes columnas fueron las encontradas con NaN's"""
# age
# time_of_service
# work_life_balance
# var2
# var4
"""para estas columnas opto por usar el por usar el promedio general de cada columna para rellenar los espacios faltantes para no borrar las filas y no perder informacion o poner ceros deliberadamente, lo cual tiene un efecto de sesgo. es util por los tipos de datos numericos que hay en las demas filas. Se deja el dtype de cada una de las columnas"""

empleados = empleados.fillna(empleados.mean(numeric_only=True))
#verificacion de que ya no hay NaN
empleados.isna().sum()

# ----------------------- 3.a Puede usar diferentes métodos de transformación, como OrdinalEncoder, OneHotEncoder, StandardEncoder, NormalizerEncoder, etc.

"""aca se realiza una separacion, de todas las columnas que son numericas o categoricas"""

#categoricas
columnas_categoricas = empleados.select_dtypes(exclude="number").columns.tolist()
print(columnas_categoricas)
"""hay 6 variables categoricas, aqui la clave es usar onehotencoder"""

onehot = OneHotEncoder(sparse_output=False)
categoricas_codificadas = onehot.fit_transform(empleados[columnas_categoricas])
nombres_onehot = onehot.get_feature_names_out(columnas_categoricas)

empleados_categoricas = pd.DataFrame(categoricas_codificadas, columns=nombres_onehot, index=empleados.index)

#numericas
columnas_numericas = empleados.select_dtypes(include="number").columns.tolist()
print(columnas_numericas)

scaler = StandardScaler()
numericas_escaladas = scaler.fit_transform(empleados[columnas_numericas])

empleados_numericas = pd.DataFrame(
    numericas_escaladas, columns=columnas_numericas, index=empleados.index
)

"""agrupo las variables escaladas y las codigicadas"""

x_df = pd.concat([empleados_numericas, empleados_categoricas], axis=1)
x = x_df.to_numpy()
x.shape

# ----------------------- 4. Selecciona el número de grupos adecuados para agrupar usando K-means:
# ----------------------- 4.a Ejecuta el algoritmo K-means con distintos números de grupos y almacena los resultados.

np.random.seed(0)

#diccionarios vacios para tener donde se guardara el agrupamiento
resultados_silhouette = {}
resultados_calinski = {}

for k in range(2, 11):
    y_pred = KMeans(n_clusters=k).fit_predict(x)
    resultados_silhouette[k] = metrics.silhouette_score(x, y_pred)
    resultados_calinski[k] = metrics.calinski_harabasz_score(x, y_pred)

print(resultados_silhouette)
print(resultados_calinski)


# ----------------------- 4.b Utiliza una estrategia para determinar el número adecuado de grupos como K-Elbow o Silhouette-Plot.

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4))

ax0.plot(list(resultados_silhouette.keys()), list(resultados_silhouette.values()), "o-")
ax0.grid(True)
ax0.set_title("silhouette")
ax0.set_xlabel("num clusters")
ax0.set_ylabel("silhouette score")

ax1.plot(list(resultados_calinski.keys()), list(resultados_calinski.values()), "o-")
ax1.grid(True)
ax1.set_title("calinski-harabasz")
ax1.set_xlabel("num clusters")
ax1.set_ylabel("calinski-harabasz score")

plt.show()

"""los dos indices bajan todo en funcion de k, si solo se leyera el numero mas alto siempre saldria (k=2). se rompe el decremento del silhouette en (k=5)"""

for k in [2, 3, 4]:
    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

    """para dejar espacio en blanco separado de una grafica"""
    ax0.set_ylim([0, len(x) + (k + 1) * 10])

    y_pred = KMeans(n_clusters=k).fit_predict(x)
    valores_silhouette = metrics.silhouette_samples(x, y_pred)

    y_lower = 10
    for i in range(k):
        valores_i = valores_silhouette[y_pred == i]
        valores_i.sort()
        tam_i = valores_i.shape[0]
        y_upper = y_lower + tam_i
        color = cm.nipy_spectral(float(i) / k)
        ax0.fill_betweenx(np.arange(y_lower, y_upper), 0, valores_i, facecolor=color, edgecolor=color)
        ax0.text(-0.05, y_lower + 0.5 * tam_i, str(i))
        y_lower = y_upper + 10

    ax0.set_title("silhouette plot, k=" + str(k))
    ax0.set_xlabel("coeficiente de silhouette")
    ax0.set_ylabel("grupo")
    ax0.axvline(x=resultados_silhouette[k], color="red", linestyle="--")

    ax1.scatter(x[:, 0], x[:, 1], c=y_pred, cmap=plt.cm.Spectral)
    ax1.set_title("grupos, k=" + str(k))

    plt.show()

"""se elige k=4: con k=4 aparece un grupo compacto de 614 empleados con attrition_rate promedio de 0.69, contra ~0.14 en los otros tres. probar con k=5 no cambia nada!: ese mismo grupo de alto riesgo sigue apareciendo casi igual (588 empleados, 0.69 de promedio), no aporta nada nuevo, asi que k=4 es mas simple"""
k = 4
etiquetas = KMeans(n_clusters=k).fit_predict(x)
metrics.silhouette_score(x, etiquetas)

# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
# -----------------------
