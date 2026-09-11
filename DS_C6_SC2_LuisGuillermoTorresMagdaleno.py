# ----------------------- Importaciones

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from sklearn.decomposition import PCA
import plotly.graph_objects as go

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
    y_pred = KMeans(n_clusters=k, n_init=10).fit_predict(x)
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

"""los dos indices bajan casi todo el tiempo conforme aumenta k, es el sesgo as pocos grupos... ; si solo se leyera el numero mas alto, siempre saldria k=2 o k=3 ya que da 0.0692 y 0.0697. de ahi en adelante baja hasta k=10, hace falta revisar las graficas silhouette-plot"""

for k in [2, 3, 4]:
    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

    """para dejar espacio en blanco separado de una grafica"""
    ax0.set_ylim([0, len(x) + (k + 1) * 10])

    y_pred = KMeans(n_clusters=k, n_init=10).fit_predict(x)
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
etiquetas = KMeans(n_clusters=k, n_init=10).fit_predict(x)
metrics.silhouette_score(x, etiquetas)

# ----------------------- # 5. Segmenta el DataFrame original creando nuevos DataFrames con los empleados separados por grupo: Crea una nueva tabla resumen con los estadísticos adecuados que describa el comportamiento de las variables por grupo.

empleados["grupo"] = etiquetas
"""para separar por grupo1,2,3,4"""
grupos = [empleados[empleados["grupo"] == numero_grupo] for numero_grupo in range(k)]

for numero_grupo in range(k):
    print("grupo", numero_grupo, ":", len(grupos[numero_grupo]), "empleados")

"""resumen estadistico por grupo: numericas se usa el promedio, para aquellas categoricas se usa la moda"""
resumen = pd.DataFrame()
resumen["variable"] = empleados.columns[:-1]

for numero_grupo in range(k):
    fila = []
    for col in resumen["variable"]:
        if col in columnas_numericas:
            fila.append(np.round(grupos[numero_grupo][col].mean(), 2))
        else:
            fila.append(grupos[numero_grupo][col].mode()[0])
    resumen["grupo " + str(numero_grupo)] = fila

resumen

# ----------------------- 6. interpretacion de los resultados obtenidos. a. Genera diferentes visualizaciones que ayuden a mostrar las características que tienen en común los empleados dentro de cada grupo.

"""barras agrupadas: promedio normalizado por variable numerica"""

resumen_num = resumen[resumen["variable"].isin(columnas_numericas)].set_index("variable")
"""para voltear filas y columnas"""
resumen_num_t = resumen_num.transpose()
resumen_num_norm = (resumen_num_t - resumen_num_t.min()) / (resumen_num_t.max() - resumen_num_t.min())

fig_barras = go.Figure()
for numero_grupo in range(k):
    fig_barras.add_trace(
        go.Bar(
            y=resumen_num_norm.loc["grupo " + str(numero_grupo)],
            x=resumen_num_norm.columns,
            name="grupo " + str(numero_grupo),
        )
    )

fig_barras.update_layout(title="promedio normalizado por variable y grupo", yaxis_title="valor normalizado")
fig_barras.show()

"""cajas y bigotes de attrition_rate por grupo, la variable clave que pide el reto"""

fig_cajas = go.Figure()
for numero_grupo in range(k):
    fig_cajas.add_trace(go.Box(y=grupos[numero_grupo]["attrition_rate"], name="grupo " + str(numero_grupo)))
fig_cajas.update_layout(title="attrition_rate por grupo", yaxis_title="attrition_rate", boxmode="group")
fig_cajas.show()

"""visualizacion con pca, se agrupo con las 47 variables (numericas + categoricas codificadas), pca solo se usa para poder ver los 4 grupos en dos dimensiones"""

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

plt.scatter(x_pca[:, 0], x_pca[:, 1], c=etiquetas, cmap=plt.cm.Spectral)
plt.title("grupos visualizados con pca")
plt.xlabel("componente 1")
plt.ylabel("componente 2")
plt.show()

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
