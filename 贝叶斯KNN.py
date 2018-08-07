from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


print ("读取数据")
df = pd.read_csv(r"C:\Users\Administrator\Desktop\vvvv.csv")
df.fillna(df.mean())
x = df.iloc[:,2:]
y = df.iloc[:,1]
# x_pca = PCA(n_components = 50).fit_transform(x)
# x=x_pca
# y=np.concatenate((np.zeros(len(y[y<3])), np.ones(len(y[y>=3]))))

print("数据读取完成")
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.3)
#最大熵
# MaxentClassifier.train(train_toks=x_train,labels=y_train)
#knn

print("knn")
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)
y_pred=neigh.predict(x_test)
# print(neigh.predict_proba([[0.9]]))
target_name = ['score 0','score 1','score 2','score 3','score 4','score 5']
print(classification_report(y_test,y_pred, target_names=target_name))
#贝叶斯模型
print("bayes")
gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_test)

# print("Number of mislabeled points out of a total %d points : %d"
#      % (iris.data.shape[0],(iris.target != y_pred).sum()))
target_name = ['score 0','score 1','score 2','score 3','score 4','score 5']
print(classification_report(y_test,y_pred, target_names=target_name))

