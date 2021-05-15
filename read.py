# 导入模块
import os
import numpy as np
import tensorflow.keras as K
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,roc_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.compat.v1.disable_eager_execution()

data1 = pd.read_excel('D:\毕业设计\数据\\HRV_data.xlsx','Sheet1',index_col=0)
data1.to_csv('D:\毕业设计\数据\\HRV_data.csv',encoding='utf-8')
pima = pd.read_csv('D:\毕业设计\数据\\HRV_data.csv') 
# pima = pd.read_csv('D:\毕业设计\数据\\HRV_data.csv') 
clo_names = pima.columns.tolist()
print(clo_names)
# to_show = clo_names[:6] + clo_names[-6:]
# pima[:].head(3)

to_drop = []
pima_01 = pima.drop(to_drop,axis=1)

# pima_01.head
clo_names_01 = pima_01.columns.tolist
print(clo_names_01)
# pima_01.shape

pima_01.describe()

from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
X = pima_01.iloc[:, 0:-2] # 特征列 0-12列，不含第13列
Y = pima_01.iloc[:, -1] # 目标列为第13列


X_features = pd.DataFrame(data = X, columns=(['BMI','高血压病史','饮酒史','SDS标准分','SAS标准分','最小呼吸','平均呼吸', '最小心率', '最大心率', '心血管健康指数']))
X_features.head()
 
Y_features = pd.DataFrame(data =Y, columns=['Y']) 
# 构造新特征DataFrame

Y_01 = pd.get_dummies(Y).values


rescaledX = StandardScaler().fit_transform(X_features)
X = pd.DataFrame(data = rescaledX, columns = X_features.columns)



from sklearn.model_selection import train_test_split

seed = 8 #重现随机生成的训练
test_size = 0.25 #33%测试，67%训练
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_01, test_size=test_size, random_state=seed)

# 在再次训练之前重置训练
from keras import backend as back
curr_session = tf.compat.v1.get_default_session()
# close current session
if curr_session is not None:
    curr_session.close()
# reset graph
back.clear_session()
# create new session
s = tf.compat.v1.InteractiveSession()
tf.compat.v1.keras.backend.set_session(s)

# 定义模型
init = K.initializers.glorot_uniform(seed=2)
# lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0
simple_adam = K.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0)
model = K.models.Sequential()
model.add(K.layers.Dense(units=16, input_dim=10, kernel_initializer=init, activation='sigmoid'))
# model.add(K.layers.Dense(units=18, kernel_initializer=init, activation='sigmoid'))
# model.add(K.layers.Dense(units=12, kernel_initializer=init, activation='sigmoid'))
# model.add(K.layers.Dense(units=24, kernel_initializer=init, activation='sigmoid'))
# model.add(K.layers.Dense(units=14, kernel_initializer=init, activation='sigmoid'))
model.add(K.layers.Dense(units=8, kernel_initializer=init, activation='sigmoid'))
# model.add(K.layers.Dense(units=8, kernel_initializer=init, activation='sigmoid'))
model.add(K.layers.Dense(units=4, kernel_initializer=init, activation='sigmoid'))
model.add(K.layers.Dense(units=2, kernel_initializer=init, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer=simple_adam, metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])

# 训练模型
b_size = 1
max_epochs = 800
print("Starting training ")
# h = model.fit(X_train, Y_train, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1)
h = model.fit(X_train, Y_train, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1, validation_data=(X_test, Y_test))
print("Training finished \n")

# 评估模型
eval = model.evaluate(X_test, Y_test, verbose=0)
print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" \
          % (eval[0], eval[1] * 100) )




# 使用模型进行预测
Y_pred = model.predict(X_test)
Y_pred

Y_pred_class = np.argmax(Y_pred, axis=1)  #其实就是记录每个数组中值最大的数的index
Y_pred_class

Y_test_class = np.argmax(Y_test, axis=1)  #其实就是记录每个数组中值最大的数的index
Y_test_class

from sklearn.metrics import classification_report
report = classification_report(Y_test_class, Y_pred_class)
print(report)

# 保存模型/加载模型
from keras.models import load_model
# model.save('model_p01')  # creates a HDF5 file 'dl_T.h5'
# del model  # 删除现有模型
# # 加载已保存模型
# model = load_model('dl_T.h5')

# print(h.history.keys())


# 绘制loss、accuracy曲线
from matplotlib import pyplot as plt 

# 绘制loss、accuracy曲线
# from matplotlib import pyplot as plt 

# epochs=range(len(h.history['accuracy']))
# plt.figure()
# plt.plot(epochs,h.history['accuracy'],'b',label='Training Accuracy')
# plt.title('Traing accuracy')
# plt.legend()

# plt.figure()
# plt.plot(epochs,h.history['loss'],'b',label='Training Loss')
# plt.title('Traing loss')
# plt.legend()
# plt.show()


# from matplotlib import pyplot as plt 

# epochs=range(len(h.history['accuracy']))
# plt.figure()
# plt.plot(epochs,h.history['accuracy'],'b',label='Training accuracy')
# plt.title('Training accuracy')
# plt.legend()

# plt.figure()
# plt.plot(epochs,h.history['loss'],'b',label='Training loss')
# plt.title('Training loss')
# plt.legend()

# plt.figure()
# plt.plot(epochs,h.history['val_accuracy'],'b',label='Testing accuracy')
# plt.title('Testing accuracy')
# plt.legend()

# plt.figure()
# plt.plot(epochs,h.history['val_loss'],'b',label='Testing loss')
# plt.title('Testing loss')
# plt.legend()

# plt.show()



lr_accuracy_score=accuracy_score(Y_test_class,Y_pred_class)
lr_preci_score=precision_score(Y_test_class,Y_pred_class)
lr_recall_score=recall_score(Y_test_class,Y_pred_class)
lr_f1_score=f1_score(Y_test_class,Y_pred_class)
lr_auc=roc_auc_score(Y_test_class,Y_pred_class)
print('lr_accuracy_score: %f,lr_preci_score: %f,lr_recall_score: %f,lr_f1_score: %f,lr_auc: %f'%(lr_accuracy_score,lr_preci_score,lr_recall_score,lr_f1_score,lr_auc))
print(metrics.cohen_kappa_score(Y_test_class,Y_pred_class))
# print("MUT_X:",metrics.confusion_matrix(Y_test_class,Y_pred_class))
# lr_fpr,lr_tpr,lr_threasholds=roc_curve(Y_test_class,Y_pred_class) # 计算ROC的值,lr_threasholds为阈值
# plt.title("roc_curve of %s(AUC=%.4f)" %('DNN',lr_auc))
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.plot(lr_fpr,lr_tpr)
# plt.show()



# acc = h.history['accuracy']
# val_acc = h.history['val_accuracy']
# loss = h.history['loss']
# val_loss = h.history['val_loss']
# epochs = range(len(acc))

# plt.plot(epochs,acc, 'b', label='Training accuracy')
# plt.plot(epochs, val_acc, 'r', label='validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend(loc='lower right')
# plt.figure()

# plt.plot(epochs, loss, 'r', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()


# # clf.fit(X_train,y_train)
# ## 绘图
# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# estimators_num=len(model.estimators_)
# X=range(1,estimators_num+1)
# ax.plot(list(X),list(model.staged_score(X_test,Y_test)),label="Testing score")
# plt.show()  


# tk pgu
# acc = h.history['accuracy']
# val_acc = h.history['val_accuracy']
# loss = h.history['loss']
# val_loss = h.history['val_loss']
# epochs = range(len(acc))

# plt.plot(epochs,acc, 'b', label='Training accuracy')
# plt.plot(epochs, val_acc, 'r', label='validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend(loc='lower right')
# plt.figure()

# plt.plot(epochs, loss, 'r', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()





