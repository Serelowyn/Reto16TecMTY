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

empleados.head()

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
# ----------------------- 
