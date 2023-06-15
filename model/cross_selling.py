import numpy as np
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori,association_rules
import  pandas as pd
import argparse
import joblib
from utils.utils import get_retail_customer_data_structure
import os
import json
from data_process.retail_customer_dataloader import load_data

#产品和编号的对应
product_info = {"存款":1,"保险":2,"基金和积利宝":3,"积存金":4,"理财":5,"国债":6,"代销资管产品":7,"贷款和信用卡":8}
_, _, _,groupby_columns,relation_columns = get_retail_customer_data_structure()
products_name = [list(i.keys())[0] for i in relation_columns]

#因时间关系，这里采用最简单的train方式
def customer_clustering(data,group_by_columns,num_clusters = 10,n_init = 10,mode="infer",model_path="./retail_customer.pkl"):

    data_for_clustering = data[groupby_columns]
    if mode == "train":
        model = KMeans(init='k-means++', n_clusters=num_clusters, n_init=n_init)
        model.fit(data_for_clustering)
        joblib.dump(model,model_path)
    else:
        model = joblib.load(model_path)
        classes = model.predict(data_for_clustering)
        data["客户类别"] = classes
        # 先将客户数据按类别划分
        classes = data["客户类别"].unique()
        customers = [data[data["客户类别"].isin([i])] for i in classes]
        return customers

def write_to(association_rule_results:dict,antecedents,filepath="../resultfiles/cross_selling.txt",mode="a+",encoding='utf-8'):
    with open(filepath,mode=mode,encoding=encoding) as f:
        for customer in customer_antecedents:
            for key in list(association_rule_results.keys()):
                if set(customer[2]) == set(json.loads(key.replace("\'", "\""))):
                    text = "客户编号:{},客户名称:{},已拥有产品:{},向其推荐产品:{},置信度:{},提升度:{}".format(customer[0],customer[1],customer[2],association_rule_results[key][0],association_rule_results[key][3],association_rule_results[key][4])
                    text = "\"" + "{}".format(customer[0]) + "\"" + "," + "\"" + ",".join([str(product_info[p]) for p in association_rule_results[key][0]]) + "\""
                    f.write(text+"\n")
"""
商品关联规则分析。Apriori算法。
支持度: 先验概率
置信度：条件概率
提升度：条件概率与先验概率的比。意思为购买商品A后购买商品B的概率与购买商品B的概率的比，称为A对B的提升。
"""
def association_rule(data,products_name,min_support=0.1,metric='confidence',min_threshold=0.3):
    #每个用户持有商品的情况(0或1表示是否持有)
    products_holding = data[["客户编号","客户名称"]+products_name]

    #频繁项集
    frequent_itemsets = apriori(products_holding[products_name], min_support=min_support, use_colnames=True)
    frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)
    frequent3 = frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x)) == 3]
    #这里我只要支持度、置信度、提升度
    columns_ordered = [
        "antecedents",
        "consequents",
        "antecedent support",
        "consequent support",
        "confidence",
        "lift",
    ]
    association_rule = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)[columns_ordered]
    association_rule.sort_values(by='lift', ascending=False, inplace=True)

    """
    这里我用antecedent_support consequent_support confidence  lift这几个值作为推荐的逻辑基础
    即：
    antecedent_support >= threshold1 and consequent_support >= threshold2 and confidence >= threshold3 and lift >= threshold4
    此时认为这条推荐是有可信度的。
    这些阈值的设定没有绝对标准，可优化。
    """
    antecedent_support_threshold = 0.1
    consequent_support_threshold = 0.1
    confidence_threshold = 0.3
    lift_threshold = 1.0

    results = {}
    #先根据阈值筛选项集
    for result in  [ [i if type(i) == float else list(i) for i in row] for row in association_rule.to_numpy()]:
        if result[2] >= antecedent_support_threshold and result[3] >= consequent_support_threshold and result[4] >= confidence_threshold and result[5] >= lift_threshold:
            if not str(result[0]) in list(results.keys()):
                results[str(result[0])] = result[1:]
            else:
                #筛选相同产品组合的antecedent中置信度最高的那个
                results[str(result[0])] = result[1:] if result[4] > results[str(result[0])][3] else results[str(result[0])]

    #利用关联信息，向每个和antecedent一致的客户推荐对应的consequent，如果某个客户的信息找不到对应的antecedent，那么就不为其推荐
    for column in products_name:
        data[column] = [column if i == 1 else i for i in list(data[column])]

    data = list(data[["客户编号","客户名称"]+products_name].to_numpy())
    customer_antecedent = []
    for cu in data:
        temp1 = [cu[0],cu[1]]
        temp2 = []
        for i in cu[2:]:
            if i != 0:
                temp2.append(i)

        temp1.append(temp2)
        customer_antecedent.append(temp1)

    return results,customer_antecedent



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path',required=True, type=str, help='.del file')
    parser.add_argument('--output_data_path', required=True, type=str, help='.txt file')
    args = parser.parse_args()
    data_filepath = args.input_data_path
    data = load_data(data_filepath)#路径不能有中文
    customers = customer_clustering(data,group_by_columns=groupby_columns)
    for customer in customers:#每类用户分别做关联规则
        association_rule_results,customer_antecedents = association_rule(customer,products_name=products_name)
        write_to(association_rule_results,customer_antecedents)