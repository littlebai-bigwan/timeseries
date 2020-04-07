import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import datetime


# 1、load data and process data
time1 = datetime.datetime.now()

df = pd.read_excel("data/weather-beijing.xlsx",usecols=["日期","气温"])
df["最高气温"] = df["气温"].map(lambda x:int(x.replace(' ','').split('\n/\n')[0]))
df["最低气温"] = df["气温"].map(lambda x:int(x.replace(' ','').split('\n/\n')[1]))
df["日期"] = df["日期"].map(lambda x:x.replace('年','-'))
df["日期"] = df["日期"].map(lambda x:x.replace('月','-'))
df["日期"] = df["日期"].map(lambda x:x.replace('日',''))
df['日期']=pd.to_datetime(df['日期'],format='%Y-%m-%d')

# 2、平稳性检验
# https://blog.csdn.net/u012735708/article/details/82460962
# 平稳性检验一般采用观察法和单位根检验法。
# 观察法：需计算每个时间段内的平均的数据均值和标准差。
# 单位根检验法：通过Dickey-Fuller Test 进行判断，大致意思就是在一定置信水平下，对于时序数据假设 Null hypothesis: 非稳定。
# 这是一种常用的单位根检验方法，它的原假设为序列具有单位根，即非平稳，对于一个平稳的时序数据，就需要在给定的置信水平上显著，拒绝原假设。


def draw_ts(timeseries):
    f = plt.figure(facecolor='white')
    timeseries.plot(color='blue')
    plt.show()

# 移动平均图
def draw_trend(timeseries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeseries.rolling(window=size).mean()
    # 对size个数据移动平均的方差
    rol_std = timeseries.rolling(window=size).std()

    timeseries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_std.plot(color='black', label='Rolling standard deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

# Dickey-Fuller test:
def teststationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput


# 平稳性检验方法1
ts = df['最高气温']
draw_trend(ts,365)
#通过上图，我们可以发现数据的移动平均值/标准差比较稳定。

# 平稳性检验方法2
print("---------------ts--------------")
print(teststationarity(ts))
# 通过p-value和Test Statistic值对比，p<Test Statistic 成立，平稳序列；
# 且通过DF的数据可以明确的看出，在95%置信度下，数据是稳定的。


# 3 时序数据的预测（用sarmiax）
diff_365 = ts.diff(365)
diff_365.dropna(inplace=True)
print("---------------diff_365-------------")
print(teststationarity(diff_365))
# 通过DF的数据可以明确的看出，在99%置信度下，数据是稳定的。

# 画图acf和pacf
def draw_acf_pacf(ts,lags):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts,ax=ax1,lags=lags)
    ax2 = f.add_subplot(212)
    plot_pacf(ts,ax=ax2,lags=lags)
    plt.subplots_adjust(hspace=0.5)
    plt.show()
draw_acf_pacf(diff_365,40)
# 根据acf和pacf，得到p=3，q=2


model1 = sm.tsa.statespace.SARIMAX(ts, order=(3,0,2),
                                              seasonal_order=(0,1,0, 365))
result_arima = model1.fit(disp=-1)
predict_ts = result_arima.predict()



testsize = int(ts.size*0.1)
ts = ts[-testsize:]
predict_ts = predict_ts[-testsize:]

ts = ts[predict_ts.index]
predict_ts.plot(color='blue', label='Predict')
ts.plot(color='red', label='Original')
plt.legend(loc='best')
plt.title('MSE: %.4f'% mean_squared_error(predict_ts,ts))
plt.show()

time2 = datetime.datetime.now()
print("weather arima takes"+str((time2-time1).seconds)+" s")

print("mse",mean_squared_error(predict_ts,ts))