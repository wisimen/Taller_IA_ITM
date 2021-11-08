import yaml
import sqlalchemy as db
import pandas as pd
from typing import NamedTuple, Dict
import json
from munch import DefaultMunch

stack_dict={}
with open('stack.yml') as f:
    stack_dict = yaml.load(f, Loader=yaml.FullLoader)
    undefined = object()
    stack_dict = DefaultMunch.fromDict(stack_dict, undefined)

db_name = stack_dict.services.db.environment.MYSQL_DATABASE
user = stack_dict.services.db.environment.MYSQL_USER
user_pass = stack_dict.services.db.environment.MYSQL_PASSWORD
port = stack_dict.services.db.environment.MYSQL_PORT


conn_string=f"mysql+pymysql://{user}:{user_pass}@localhost:{port}/{db_name}?charset=utf8mb4"
print(conn_string)
engine = db.create_engine(conn_string)
connection = engine.connect()
metadata = db.MetaData()
frutas = db.Table('frutas', metadata, autoload=True, autoload_with=engine)
tipo_frutas = db.Table('tipo_frutas', metadata, autoload=True, autoload_with=engine)

query = db.select([frutas.columns.id_tipo_fruta, tipo_frutas.columns.idtipo_frutas])
results = connection.execute(query).fetchall()
df = pd.DataFrame(results)
df.columns = results[0].keys()
df.head(5)

print(df)
# INSERT INTO `fruits_db`.`frutas` (`ruta`, `is_test`, `id_tipo_fruta`) VALUES ('ab', b'1', b'1');

