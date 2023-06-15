#-*- coding:utf-8 -*-
from data_process.customer_losing_dataloader import load_data
import argparse
from sklearn import linear_model
import joblib
import json
import numpy as np

#简单实现
def prediction(data,mode="infer",model_path="./customer_losing.pkl",output_path="../resultfiles/customer_losing.txt"):
    # 拿到真实数据后训练模型请用特征分析选择合适的特征。这里未做
    column_for_train = ['员工人数', '企业规模', '开户年限', '企业名称变更次数', '经营范围变更次数', '经营地址变更次数',
                        '本月日均存款余额', '活期存款月日均', '通知存款月日均', '协议存款月日均', '保证金存款月日均',
                        '“定享通”定期存款月日均', '大额存单月日均', '结构性存款月日均', '本月日均贷款余额',
                        '本月日均保本理财余额', '本月日均非保本理财余额', '本月日均贴现余额', '本月日均开票余额',
                        '园区通交易笔数', '园区通交易金额', '银校通交易笔数', '银校通交易金额', '教培系统交易笔数',
                        '教培系统交易金额', '单位结算卡笔数', '单位结算卡金额', 'POS收单笔数', 'POS收单金额',
                        '对公ETC交易次数', '对公ETC交易金额', '对公银银转账交易次数', '对公银银转账交易金额',
                        '代缴社保笔数', '代缴社保金额', '代缴公积金笔数', '代缴公积金金额', '企业网银交易笔数',
                        '企业网银交易金额', '柜面交易笔数', '柜面交易金额']
    data_train_df = data[column_for_train]
    data_train_np = data_train_df.to_numpy()
    if mode == "train":
        #拿到真实数据后请做特征工程。这里未做
        labels = np.array(list(data["是否流失"]))
        #拿到真实数据后请做数据集划分，这里未做
        model = linear_model.LogisticRegression(solver='liblinear', C=100)
        #正式训练时请追踪指标，这里未做
        model.fit(data_train_np,labels)
        joblib.dump(model, model_path)
    else:
        model = joblib.load(model_path)
        results = model.predict(data_train_np)
        data["预测结果"] = results
        data = data[["客户编号","预测结果"]].to_numpy()
        with open(output_path,mode="w",encoding='utf-8') as f:
            for row in data:
                if row[1] == 1:
                    f.write("\""+str(row[0])+"\""+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path',required=True, type=str, help='.del file')
    parser.add_argument('--output_data_path', required=True, type=str, help='.txt file')
    args = parser.parse_args()
    data = load_data(args.input_data_path)
    prediction(data=data,output_path=args.output_data_path)