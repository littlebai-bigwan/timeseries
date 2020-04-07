import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

# 1、load data
df = pd.read_csv('data/passage.csv')
df.columns = ["month","number"]
df['month'] = pd.to_datetime(df['month'])
df.index = df['month']
df.drop(['month'], axis=1, inplace=True)

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
ts = df['number']
draw_trend(ts,12)
#通过上图，我们可以发现数据的移动平均值/标准差有越来越大的趋势，是不稳定的。

# 平稳性检验方法2
print("---------------ts--------------")
print(teststationarity(ts))
# 通过p-value和Test Statistic值对比，p<Test Statistic 不成立，非平稳序列；
# 且通过DF的数据可以明确的看出，在任何置信度下，数据都不是稳定的。

# 3、转化为平稳序列
# 数据不稳定的原因主要有以下两点：
# 趋势（trend）-数据随着时间变化。比如说升高或者降低。
# 季节性(seasonality)-数据在特定的时间段内变动。比如说节假日，或者活动导致数据的异常。

# 转化方法1 对数
ts_log = np.log(ts)
draw_ts(ts_log)

# 转化方法2 移动平均
def draw_moving(timeSeries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean=timeSeries.ewm(halflife=size,min_periods=0,adjust=True,ignore_na=False).mean()

    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()

draw_moving(ts_log, 12)

# 转化方法3 差分
diff_12 = ts_log.diff(12)
diff_12.dropna(inplace=True)
diff_12_1 = diff_12.diff(1)
diff_12_1.dropna(inplace=True)
print("---------------diff_12_1--------------")
print(teststationarity(diff_12_1))
#从上面的统计检验结果可以看出，经过对数，12阶差分和1阶差分后，该序列满足平稳性的要求了。


# 转化方法4 分解
def decompose(timeseries):
    # 返回包含三个部分 trend（趋势部分） ， seasonal（季节性部分） 和residual (残留部分)
    decomposition = seasonal_decompose(timeseries)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    return trend, seasonal, residual

trend, seasonal, residual = decompose(ts_log)
residual.dropna(inplace=True)
draw_trend(residual, 12)
print("---------------residual--------------")
print(teststationarity(residual))
# 如图所示，数据的均值和方差趋于常数，几乎无波动(看上去比之前的陡峭，但是要注意他的值域只有[-0.05,0.05]之间)，所以直观上可以认为是稳定的数据。
# 另外DFtest的结果显示，Statistic值原小于1%时的Critical value，所以在99%的置信度下，数据是稳定的。

# 4 时序数据的预测一（用armia）如下方法参考https://blog.csdn.net/u012735708/article/details/82460962
rol_mean = ts_log.rolling(window=12).mean()
rol_mean.dropna(inplace=True)
ts_diff_1 = rol_mean.diff(1)
ts_diff_1.dropna(inplace=True)
print("---------------ts_diff_1--------------")
print(teststationarity(ts_diff_1))

ts_diff_2 = ts_diff_1.diff(1)
ts_diff_2.dropna(inplace=True)
print("---------------ts_diff_2--------------")
print(teststationarity(ts_diff_2))

# 画图acf和pacf
def draw_acf_pacf(ts,lags):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts,ax=ax1,lags=lags)
    ax2 = f.add_subplot(212)
    plot_pacf(ts,ax=ax2,lags=lags)
    plt.subplots_adjust(hspace=0.5)
    plt.show()
draw_acf_pacf(ts_diff_2,30)
# 根据acf和pacf，得到p=1，q=1


model = ARIMA(ts_diff_2, order=(1,0,1))
result_arima = model.fit( disp=-1)
predict_ts = result_arima.predict()

# 一阶差分还原
diff_shift_ts = ts_diff_1.shift(1)
diff_recover_1 = predict_ts.add(diff_shift_ts)
# 再次一阶差分还原
rol_shift_ts = rol_mean.shift(1)
diff_recover = diff_recover_1.add(rol_shift_ts)


# 移动平均还原
rol_sum = ts_log.rolling(window=11).sum()
rol_recover = diff_recover*12 - rol_sum.shift(1)
# 对数还原
log_recover = np.exp(rol_recover)
log_recover.dropna(inplace=True)

# 测试集的评估
testsize = int(ts.size*0.3)
log_recover = log_recover[-testsize:]
predict_ts = predict_ts[-testsize:]

ts = ts[log_recover.index]  # 过滤没有预测的记录plt.figure(facecolor='white')
log_recover.plot(color='blue', label='Predict')
ts.plot(color='red', label='Original')
plt.legend(loc='best')
plt.title('MSE: %.4f'% mean_squared_error(log_recover,ts))
plt.show()


# 4、时间序列预测二（用sarimax，将季节性因素考虑到模型中）
ts = df['number']
ts_log = np.log(ts)
model1 = sm.tsa.statespace.SARIMAX(ts_log, order=(1,2,1),
                                              seasonal_order=(0,1,0, 12))
result_arima = model1.fit(disp=-1)
predict_ts = result_arima.predict()

# 对数还原
log_recover = np.exp(predict_ts)
log_recover.dropna(inplace=True)

testsize = int(ts.size*0.3)
log_recover = log_recover[-testsize:]
predict_ts = predict_ts[-testsize:]

ts = ts[log_recover.index]
log_recover.plot(color='blue', label='Predict')
ts.plot(color='red', label='Original')
plt.legend(loc='best')
plt.title('MSE: %.4f'% mean_squared_error(log_recover,ts))
plt.show()

