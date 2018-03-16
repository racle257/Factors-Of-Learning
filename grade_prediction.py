import pandas as pd
import xgboost as xgb

train_data = pd.read_csv('data/midData/train_data_college.csv',header=0,encoding='GBK')
test_data = pd.read_csv('data/midData/test_data_high_school.csv',header=0,encoding='GBK')

dataset1_y = train_data['grade_gaokao']
dataset1_x = train_data.drop(['id','grade_gaokao'],axis=1)
prediction = test_data[['id']]
dataset2_x = test_data.drop(['id'],axis=1)

params={'booster':'gbtree',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.65,
	    'colsample_bytree':0.65,
	    'colsample_bylevel':0.65,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }
#
# #参数测试
# #训练集再划分：0.9训练，0.1测试，输出模型的MSE
# from sklearn.cross_validation import train_test_split
# from sklearn import metrics
# x_train, x_test, y_train, y_test = train_test_split(dataset1_x, dataset1_y, test_size = 0.1)
# train_model = xgb.DMatrix(x_train,label=y_train)
# test_model = xgb.DMatrix(x_test,label=y_test)
# watchlist = [(train_model,'train')]
# model = xgb.train(params,train_model,num_boost_round=500,evals=watchlist)
# preds = model.predict(test_model)
# print(metrics.mean_squared_error(test_model.get_label(), preds))



#预测
dataset1 = xgb.DMatrix(dataset1_x,label=dataset1_y)
dataset2 = xgb.DMatrix(dataset2_x)
watchlist = [(dataset1,'train')]
model = xgb.train(params,dataset1,num_boost_round=500,evals=watchlist)
prediction['label'] = model.predict(dataset2)
prediction['label'] = prediction['label'].astype('int')
prediction = prediction.sort_values(["label"],ascending=False)
prediction.to_csv("data/resultData/xgb_preds.csv",index=None,header=None,encoding='GBK')
# print(prediction.info())
#save feature score
feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
fs = []
total = 0
for (key,value) in feature_score:
    total = total + value
for (key,value) in feature_score:
    fs.append("{0},{1},{2:.2f}%\n".format(key,value,value/total*100))
with open('data/resultData/xgb_feature_score.csv','w') as f:
    f.writelines("feature,score,rate\n")
    f.writelines(fs)