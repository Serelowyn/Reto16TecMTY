# ----------------------- Importaciones

import pandas as pd

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
"""para estas columnas opto por usar el por usar el promedio general de cada columna para rellenar los espacios faltantes para no borrar las filas y no perder informacion o poner ceros deliberadamente, lo cual tiene un efecto de sesgo. es util por los tipos de datos numericos que hay en las demas filas."""

empleados = empleados.fillna(empleados.mean(numeric_only=True))
#verificacion de que ya no hay NaN
empleados.isna().sum()

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
