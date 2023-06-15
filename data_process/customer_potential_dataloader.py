import pandas as pd
from utils.logger_module import get_logger
from utils.utils import certain_row,drop_the_none_row,get_customer_potential_co

logger = get_logger("../logs/customer_potential.txt")

#使用到两个数据文件
def load_data(filepath1,filepath2):
    column_potential_customer,column_relation = get_customer_potential_co()
    data_potential_customer = pd.read_csv(filepath1,header=None)
    data_relation = pd.read_csv(filepath2,header=None)
    data_potential_customer.columns  = column_potential_customer
    data_relation.columns = column_relation

    return drop_the_none_row(data_potential_customer),drop_the_none_row(data_relation)



if __name__ == "__main__":
    load_data("./potential_customer.del","./potential_customer_relation.del")

