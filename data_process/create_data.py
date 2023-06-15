#没有数据，自己构造数据
from utils.utils import get_retail_customer_data_structure
import pandas as pd
import random

columns,column_type,column_value,_,_ = get_retail_customer_data_structure()
#拟造数据不考虑字段之间的逻辑关联性

def create_retail_customer_data(rows_count:int,columns:list,column_type:list,column_value:list,output_filename:str=None)->list:
    row_data = []  # 每个子元素是一行数据
    for i in range(rows_count):
        temp_row_data = []
        for j,column in enumerate(columns):
            temp_row_data.append(column_value[j][random.randint(0,len(column_value[j])-1)])
        row_data.append(temp_row_data)

    df = pd.DataFrame(row_data,columns=columns)
    df.to_csv(output_filename,index=0,header=0)


if __name__ == "__main__":
    create_retail_customer_data(10000,columns,column_type,column_value,"./retail_customer.del")
