import yaml
import sqlalchemy as db
import pandas as pd
from munch import DefaultMunch

class DbConnection:
    frutas= {}
    tipo_frutas= {}
    def __init__(self):
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
        self.connection = engine.connect()
        metadata = db.MetaData()

        # Definir Tablas
        self.frutas = db.Table('frutas', metadata, autoload=True, autoload_with=engine)
        self.tipo_frutas = db.Table('tipo_frutas', metadata, autoload=True, autoload_with=engine)

    def get_fruits(self, is_test= False):
        query = db.select([
            self.frutas.columns.ruta,
            self.frutas.columns.id_tipo_fruta
            ]).join(self.tipo_frutas).where(self.frutas.columns.is_test == is_test)
        results = self.connection.execute(query).fetchall()
        df = pd.DataFrame(results)
        # if(is_test):
        #     df = df.head(100)
        return df

    def get_labels(self):
        query = db.select([
            self.tipo_frutas.columns.idtipo_frutas,
            self.tipo_frutas.columns.nombre
            ])
        results = self.connection.execute(query).fetchall()
        df = pd.DataFrame(results)
        return df