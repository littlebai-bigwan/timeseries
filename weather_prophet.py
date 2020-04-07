# https://blog.csdn.net/anshuai_aw1/article/details/83412058

import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

# 1、load data
df = pd.read_csv('data/weather-beijing.csv')
df.columns = ['ds','y']
df.plot()

# 2、通过图像，发现2016-01-23数据为异常值，删除异常值
df.loc[(df['ds'] == '2016-01-23'), 'y'] = None
df.plot()

# 3、训练集、测试集划分
trainsize = int(df.shape[0]*0.9)
train = df[0:trainsize]
test = df[trainsize:df.shape[0]]

# 4、模型训练
# changepoint_prior_scale默认为0.05,增大changepoint_prior_scale表示让模型拟合更多，可能发生过拟合风险
model = Prophet(changepoint_prior_scale=0.5)
model.fit(df);

#5、模型预测
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

#6、模型评估
print("mse is",mean_squared_error(test['y'].values,forecast['yhat'].values[trainsize:df.shape[0]]))
