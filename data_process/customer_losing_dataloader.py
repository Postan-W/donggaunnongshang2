import pandas as pd
from utils.logger_module import get_logger
from utils.utils import certain_row,get_customer_losing_co,drop_the_none_row

logger = get_logger("../logs/customer_losing.txt")

def load_data(filepath):
    data_without_header = pd.read_csv(filepath, header=None)  # 客户输入的数据不带表头
    columns,useful_columns = get_customer_losing_co()
    data_without_header.columns = columns
    data = data_without_header
    data_useful = data[useful_columns]

    #流失客户定义：客户当期金融资产月日均较近三月下降比例超过50%
    current_month = list(data_useful["金融资产月日均"])
    last_3_month = list(data_useful["近三个月金融资产月日均"])
    label = []
    for i in range(len(data_useful)):
        label.append(1 if current_month[i]/last_3_month[i] < 0.5 else 0)
    data_useful["是否流失"] = label

    return drop_the_none_row(data_useful)

if __name__ == "__main__":
    print(load_data("customer_losing.del"))
