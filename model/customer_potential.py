#-*- coding:utf-8 -*-
from data_process.customer_potential_dataloader import load_data
import argparse
from sklearn import linear_model
import joblib
import json
import numpy as np

#模型方面请完善。这里是简单调用一个分类模型进行训练或推理
def prediction(data_potential,data_relation,mode="infer",model_path="./customer_potential.pkl",output_path="../resultfiles/customer_potential.txt"):
    #注意：这里并没有用到关联数据表中的数据。
    column_for_train = ['近1个月交易金额','近3个月交易金额','近1年交易金额','交易笔数','近1个月交易笔数','近3个月交易笔数','近1年交易笔数']
    data_for_train = data_potential[column_for_train]
    if mode == "train":
        month_average_count = list(data_potential['月均交易笔数'])
        month_average_money = list(data_potential['月均交易金额'])
        # 潜客定义：月均交易笔数大于3笔或月均交易金额大于300万元
        labels = []
        for i in range(len(data_potential)):
            labels.append(1 if month_average_count[i] > 3 or month_average_money[i] > 3000000 else 0)
        model = linear_model.LogisticRegression(solver='liblinear', C=100)
        model.fit(data_for_train.to_numpy(), labels)
        joblib.dump(model, model_path)
    else:
        model = joblib.load(model_path)
        results = model.predict(data_for_train.to_numpy())
        position = data_potential[["客户编号","注册地址","经营地址"]].to_numpy()

        with open(output_path,mode="w",encoding="utf-8") as f:
            for i,result in enumerate(results):
                if result == 1 and ("广东" in position[i][1] or "广东" in position[i][2]):
                    f.write("\"" + str(position[i][0]) + "\"" + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path1',required=True, type=str, help='.del file')
    parser.add_argument('--input_data_path2', required=True, type=str, help='.del file')
    parser.add_argument('--output_data_path', required=True, type=str, help='.txt file')
    args = parser.parse_args()
    data1,data2 = load_data(args.input_data_path1,args.input_data_path2)
    prediction(data1,data2,output_path=args.output_data_path)