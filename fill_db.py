import yaml
import sqlalchemy as db
import pandas as pd
from munch import DefaultMunch
import os

# Abrir archivo de configuración
stack_dict={}
with open('stack.yml') as f:
    stack_dict = yaml.load(f, Loader=yaml.FullLoader)
    undefined = object()
    stack_dict = DefaultMunch.fromDict(stack_dict, undefined)

# Obtener variables del archivo de configuración
db_name = stack_dict.services.db.environment.MYSQL_DATABASE
user = stack_dict.services.db.environment.MYSQL_USER
user_pass = stack_dict.services.db.environment.MYSQL_PASSWORD
port = stack_dict.services.db.environment.MYSQL_PORT

# Crear conexión a la base de datos
conn_string=f"mysql+pymysql://{user}:{user_pass}@localhost:{port}/{db_name}?charset=utf8mb4"
print(conn_string)
engine = db.create_engine(conn_string)
connection = engine.connect()
metadata = db.MetaData()

# Definir Tablas
frutas = db.Table('frutas', metadata, autoload=True, autoload_with=engine)
tipo_frutas = db.Table('tipo_frutas', metadata, autoload=True, autoload_with=engine)

# Recorrer carpeta de frutas
for root, dirs, files in os.walk('fruits'):
    # para cada imagen de fruta encontrada realizar inserción
    for imagen in files:
        # Verificar que sea una imágen para no tomar otro tipo de archivos
        if imagen.lower().endswith('.jpg'):
            ruta = root.split("\\")

            # obtener el tipo del nombre de la carpeta
            tipo_fruta = ruta[-1]

            # definir función de búsqueda de codigo de fruta
            def consultar_tipo_fruta():
                query = db.select([tipo_frutas.columns.idtipo_frutas,tipo_frutas.columns.nombre]).where(tipo_frutas.columns.nombre == tipo_fruta)
                results = connection.execute(query).fetchall()
                df = pd.DataFrame(results)
                return df
            # Consultamos el tipo de fruta
            df = consultar_tipo_fruta()
            if  df.size == 0:
                #Si no exite lo creamos y lo consultamos
                stmt = (db.insert(tipo_frutas).values(nombre=tipo_fruta))
                connection.execute(stmt)
                df = consultar_tipo_fruta()

            # tomar datos para inserción de imagen
            idtipo_frutas = df[0][0]
            is_test = ruta[-2]=="Test"
            ruta_img= root+ "\\" + imagen

            # definir función de búsqueda de imagen
            def consultar_imagen():
                query = db.select([
                    frutas.columns.ruta,
                    frutas.columns.is_test,
                    tipo_frutas.columns.idtipo_frutas,
                    tipo_frutas.columns.nombre
                    ]).where(frutas.columns.ruta == ruta_img)
                results = connection.execute(query).fetchall()
                df = pd.DataFrame(results)
                return df

            # Consultamos la imagen de la BD
            df = consultar_imagen()
            if  df.size == 0:
                stmt = (db.insert(frutas).values(ruta=ruta_img,is_test=is_test,id_tipo_fruta= idtipo_frutas))
                connection.execute(stmt)
                df = consultar_imagen()

            #Mostrar información de imagen
            print("Imagen")
            print(df)