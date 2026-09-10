# ----------------------- Importaciones

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

# ----------------------- 3.a Transforma los datos según las características de las variables:

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
# -----------------------
# -----------------------
# -----------------------
