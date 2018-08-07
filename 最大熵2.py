# -*- coding: utf-8 -*-
from nltk.classify import MaxentClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib
print ("读取数据")
df = pd.read_csv(r"E:\shixun\vvvv.csv")
df.fillna(df.mean())
x = df.iloc[:,2:]
y = df.iloc[:,1]
print("数据读取完成")
x["index"] = range(len(x))
#xdic=x.set_index("index").T.to_dict("list")
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.3)
train=[]
for i in range(len(x_train)):
    sss={}
    #sss[x.iloc[1,400]]=x.iloc[1,:400].tolist()
    # sss[x.iloc[1,400]]=x.iloc[1,:400].sum()
    sss[x_train.iloc[1, 400]]=tuple(x_train.iloc[1, :400].tolist())
    #必须是数字，而不能是列表
    train.append((sss,y_train[x_train.index[i]]))
# for i in range(len(xdic)):
#     train.append((xdic[i],y[i]))
classifier = MaxentClassifier.train(train, 'IIS', trace=0, max_iter=1000)
#from sklearn.externals import joblib
#joblib.dump(clf, "train_model.m")
# clf = joblib.load(r"C:\Users\Administrator\PycharmProjects\untitled\shixun\旅游数据处理\clf.m")

test=[]
for i in range(len(x_test)):
    sss={}
    #sss[x.iloc[1,400]]=x.iloc[1,:400].tolist()
    # sss[x.iloc[1,400]]=x.iloc[1,:400].sum()
    sss[x_test.iloc[1, 400]]=tuple(x_test.iloc[1, :400].tolist())
    #必须是数字，而不能是列表
    test.append((sss,y_test[x_test.index[i]]))
classify_results=[]
for i in test:
    classify_results.append(classifier.classify(i[0]))

target_name = ['score 0','score 1','score 2','score 3','score 4','score 5']
print(classification_report(y_test,classify_results, target_names=target_name))