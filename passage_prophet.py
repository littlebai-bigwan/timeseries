# https://blog.csdn.net/anshuai_aw1/article/details/83412058

import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

# 1、load data
df = pd.read_csv('data/passage.csv')
df.columns = ['ds','y']
df.plot()

# 2、训练集、测试集划分
trainsize = int(df.shape[0]*0.9)
train = df[0:trainsize]
test = df[trainsize:df.shape[0]]

# 3、模型训练
# 基于时间序列图形，选择乘法模型，seasonality_mode='multiplicative'
model = Prophet(seasonality_mode='multiplicative')
model.fit(df);

# 4、模型预测
# 构建待预测日期数据框，periods = 365 代表除历史数据的日期外再往后推 365 天
future = model.make_future_dataframe(periods=df.shape[0]-trainsize)
print("-----------------future.tail-----------------")
print(future.tail())

# 预测数据集
forecast = model.predict(future)
print("-----------------forcast tail-----------------")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# 展示预测结果
model.plot(forecast);

# 预测的成分分析绘图，展示预测中的趋势、周效应和年度效应
model.plot_components(forecast);

print("-----------------forcast.columns-----------------")
print(forecast.columns)

# 5、模型评价
print("mse is",mean_squared_error(test['y'].values,forecast['yhat'].values[trainsize:df.shape[0]]))
