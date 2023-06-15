import json
import pandas as pd
from utils.logger_module import get_logger
from utils.utils import certain_row,get_retail_customer_data_structure,drop_the_none_row

logger = get_logger("../logs/retail_customer.txt")


def load_data(filepath:str)->pd.DataFrame:
    data_without_header = pd.read_csv(filepath,header=None) #客户输入的数据不带表头
    columns,column_type,value_list,groupby_columns,relation_columns = get_retail_customer_data_structure()
    data_without_header.columns = columns
    data = data_without_header

    #记录空值情况
    for column in list(data.columns):
        none_index = []
        for i,v in enumerate(list(data[column].isnull())):
            if v:
                none_index.append(i+1)
        if not len(none_index) == 0:
            logger.warning("列:{},含有空值的行号为:{}".format(column, none_index))


    data = drop_the_none_row(data)
    #将要使用到的字符列数值化，依据是该列取值列表的各个值的索引
    for i,column in enumerate(list(data.columns)):
        if (column in groupby_columns) and (column_type[i] == "str"):
            temp_column_value = []
            for v in list(data[column]):
                if not v in value_list[i]:
                    logger.error("{}不在取值列表{}中".format(v,value_list[i]))
                    return
                else:
                    temp_column_value.append(value_list[i].index(v))

            data[column] = temp_column_value

    #根据各个余额情况确定8个产品的持有情况，余额为0为不持有，非0为持有，历史余额不作为判断依据。其中贷款和信用卡视为一个产品，二者余额有一个不为0就视为持有，均为0为不持有
    for product in relation_columns:
        product_name = list(product.keys())[0]
        rely_on = [list(data[column]) for column in product[product_name]]

        temp_new_column_data = []
        for i in range(len(data)):
            flag = False
            for j in range(len(rely_on)):
                if rely_on[j][i] > 0:
                    flag = True
                    break
            if flag:
                temp_new_column_data.append(1)
            else:
                temp_new_column_data.append(0)
        data[product_name] = temp_new_column_data


    return data





if __name__ == "__main__":
    load_data("./retail_customer.del")




