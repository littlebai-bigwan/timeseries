import pandas as pd
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Model, load_model

# 1、load data
df = pd.read_csv('data/passage.csv')
df.columns = ["month","number"]
df['month'] = pd.to_datetime(df['month'])
df.index = df['month']
df.drop(['month'], axis=1, inplace=True)

# 2、process data
pyplot.plot(df['number'])
pyplot.show()


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(np.reshape(df['number'].values,(df.shape[0],1)))

# according to step, create dataset
def create_dataset(dataset, step):
    dataX, dataY = [], []
    for i in range(len(dataset) - step - 1):
	    a = dataset[i:(i + step), 0]
	    dataX.append(a)
	    dataY.append(dataset[i + step, 0])
    return np.array(dataX), np.array(dataY)

step = 36
dataX, dataY = create_dataset(dataset, step)

# split train and test dataset
# 为什么不用train_test_split
# X_train, X_test, Y_train, Y_test= train_test_split(dataX, dataY, test_size=0.2, random_state=0)

test_size = int(dataX.shape[0]*0.3)
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
model = Sequential()
model.add(SimpleRNN(24))
model.add(Dropout(0.5))  # dropout层防止过拟合
model.add(Dense(1))      # 全连接层
model.add(Activation('sigmoid'))  #激活层
model.compile(optimizer=RMSprop(), loss='mse')
model.fit(X_train, Y_train, nb_epoch=50, batch_size=4, verbose=2)
model.save('passager.h5')
# model = load_model('passager.h5')

# 4、model predict
Y_predict = model.predict(X_test)

Y_predict = scaler.inverse_transform(Y_predict)
np.reshape(Y_test, (Y_test.shape[0], 1))
Y_test = scaler.inverse_transform(Y_test)

Y_predict = np.reshape(Y_predict,(Y_predict.shape[0],))
Y_test = np.reshape(Y_test,(Y_test.shape[0],))

# 5、model evaluation
print("model mean squared error is " + str(mean_squared_error(Y_test, Y_predict)))


# plot data
pyplot.plot(Y_predict)
pyplot.plot(Y_test)
pyplot.show()

