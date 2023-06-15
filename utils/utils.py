import pandas as pd
import random
from configparser import ConfigParser
import json

def get_customer_potential_co():
    conf = ConfigParser()
    conf.read("../utils/columns.ini", encoding="utf-8")
    column_potential_customer = conf["customer_potential"]["column1"]
    column_potential_customer = json.loads(column_potential_customer.replace("\'", "\""))
    column_relation = conf["customer_potential"]["column2"]
    column_relation = json.loads(column_relation.replace("\'", "\""))

    return column_potential_customer,column_relation


def get_retail_customer_data_structure():
    conf = ConfigParser()
    conf.read("../utils/columns.ini", encoding="utf-8")
    columns = conf["cross_selling"]["columns"]
    columns = json.loads(columns.replace("\'", "\""))  #json.loads列表时用双引号识别元素
    columns_name = [i["name"] for i in columns]
    column_type = [i["column_type"] for i in columns]
    value_list = [i["value_list"] for i in columns]

    groupby_columns = conf["cross_selling"]["groupby"]
    groupby_columns = json.loads(groupby_columns.replace("\'", "\""))

    relation_columns = conf["cross_selling"]["relation"]
    relation_columns = json.loads(relation_columns.replace("\'", "\""))

    return columns_name,column_type,value_list,groupby_columns,relation_columns

# get_retail_customer_data_structure()

def get_customer_losing_co():
    conf = ConfigParser()
    conf.read("../utils/columns.ini", encoding="utf-8")
    columns = conf["customer_losing"]["columns"]
    columns = json.loads(columns.replace("\'", "\""))
    useful_columns = conf["customer_losing"]["useful_columns"]
    useful_columns = json.loads(useful_columns.replace("\'", "\""))
    return columns,useful_columns



def certain_row(df:pd.DataFrame):#取一行数据，方便查看数据情况
    row_data = {}
    data_columns = list(df.columns)
    certain_row = list(df.loc[random.randint(0,df.shape[0]-1)])
    for key,value in zip(data_columns,certain_row):
        row_data[key] = value

    return row_data

drop_the_none_row = lambda df:df.dropna(axis=0,how='any')

