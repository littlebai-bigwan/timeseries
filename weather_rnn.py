import pandas as pd
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Model, load_model

# 1、load data
df = pd.read_excel("data/weather-beijing.xlsx",usecols=["日期","气温"])

# 2、process data
df["最高气温"] = df["气温"].map(lambda x:int(x.replace(' ','').split('\n/\n')[0]))
df["最低气温"] = df["气温"].map(lambda x:int(x.replace(' ','').split('\n/\n')[1]))
df["日期"] = df["日期"].map(lambda x:x.replace('年','-'))
df["日期"] = df["日期"].map(lambda x:x.replace('月','-'))
df["日期"] = df["日期"].map(lambda x:x.replace('日',''))
df['日期']=pd.to_datetime(df['日期'],format='%Y-%m-%d')

# plot data
df['最高气温'].plot()
# df['最低气温'].plot()
pyplot.show()

# plot 2020年 data
# df['最高气温'][df.__len__()-83:df.__len__()].plot()
# df['最低气温'][df.__len__()-83:df.__len__()].plot()
# pyplot.show()

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(np.reshape(df['最高气温'].values,(df.shape[0],1)))

# according to step, create dataset
def create_dataset(dataset, step):
    dataX, dataY = [], []
    for i in range(len(dataset) - step - 1):
	    a = dataset[i:(i + step), 0]
	    dataX.append(a)
	    dataY.append(dataset[i + step, 0])
    return np.array(dataX), np.array(dataY)

# step为根据前step天的数据来预测后一天的数据
step = 10
dataX, dataY = create_dataset(dataset, step)

# split train and test dataset
# 为什么不用train_test_split？时间序列数据中应该用前期的数据训练，后期数据预测。
# X_train, X_test, Y_train, Y_test= train_test_split(dataX, dataY, test_size=0.2, random_state=0)

test_size = int(dataX.shape[0]*0.1)
train_size = dataX.shape[0]-test_size
X_train = dataX[0:train_size,]
X_test = dataX[train_size:dataX.shape[0],]
Y_train = dataY[0:train_size,]
Y_test = dataY[train_size:dataX.shape[0],]

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
Y_train=np.reshape(Y_train, (Y_train.shape[0], 1))
Y_test= np.reshape(Y_test, (Y_test.shape[0], 1))


# 3、model train
# 1）第一次训练需要训练模型并保存模型
model = Sequential()
model.add(SimpleRNN(64))
model.add(Dropout(0.5))  # dropout层防止过拟合
model.add(Dense(1))      # 全连接层
model.add(Activation('sigmoid'))  #激活层
model.compile(optimizer=RMSprop(), loss='mse')
model.fit(X_train, Y_train, nb_epoch=50, batch_size=4, verbose=2)
model.save('weather.h5')
# 2）若模型已经训练好，可以直接加载模型
# model = load_model('weather.h5')

# 4、model predict
Y_predict = model.predict(X_test)

# 5、model evaluation
Y_predict = scaler.inverse_transform(Y_predict)
np.reshape(Y_test, (Y_test.shape[0], 1))
Y_test = scaler.inverse_transform(Y_test)

Y_predict = np.reshape(Y_predict,(Y_predict.shape[0],))
Y_test = np.reshape(Y_test,(Y_test.shape[0],))

print("model mean squared error is " + str(mean_squared_error(Y_test, Y_predict)))


# plot data
pyplot.plot(Y_predict)
pyplot.plot(Y_test)
pyplot.show()


# df.to_csv("data/weather-beijing.csv",encoding='utf-8',columns=["日期","最高气温"],index=False)
