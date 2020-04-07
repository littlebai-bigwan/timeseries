library(forecast)
library(tseries)

#1 load datas
trainsize = 3370*0.9
testsize = 3370*0.1
data = read.csv("/Users/jessica/PycharmProjects/machinelearning/dataanalysis/data/weather-beijing.csv",nrows=trainsize)
testdata = read.csv("/Users/jessica/PycharmProjects/machinelearning/dataanalysis/data/weather-beijing.csv",skip=trainsize)
tempseries=ts(data[2],frequency = 365)
plot.ts(tempseries)
tsdisplay(tempseries,xlab="time",ylab="temperature",lag.max = 40)

#2 对时间序列进行拆分，得到三个组件：seasonal，trend，random，可以用加法模型处理季节性
tempcomp <- decompose(tempseries)
plot(tempcomp)
#从图中看出来，具有明显的季节性因素。

#3 平稳性检验：
adf.test(tempseries)
# 通过Dickey-Fuller,可以看出序列平稳

# 取出季节性因素
tsdisplay(tempseries-tempcomp$seasonal,xlab="time",ylab="temperature",lag.max = 40)
# 非季节性p=1,q=9

# 对季节性进行差分
diff1 <- diff(tempseries,365)
plot(diff1)
tsdisplay(diff1,xlab="time",ylab="diff1",lag.max = 40)
# 非季节性p=1,q=9


# 根据acf和pacf图形，arima（1,0,9)(0,1,0)[365]
auto.arima(tempseries,trace=T)
# 根据auto.arima，arima（1,0,3)(0,1,0)[365]

# arima模型  arima(x,order=,include.mean=,method=,transform.pars=,fixed=,seasonal=)
# -x:要进行模型拟合的序列名字。
# -order:指定模型阶数。
# -include.mean:指定是否需要拟合常数项。
# -method:指定参数估计方法。
# -transform.pars:指定是否需要人为干预参数。
# -fixed:对疏系数模型指定疏系数的位置。
# -seasonal:指定季节模型的阶数与季节周期，该选项的命令格式为：
# seasonal = list(order=c(P,D,Q),period = pi）
#                 (1)加法模型：P=0,Q=0
#                 (2)乘法模型：P,Q不全为零
fit1 <- Arima(tempseries, order=c(1,0,3),seasonal=list(order=c(0,1,0),period=365))
#结果：AIC=14581.06   AICc=14581.08   BIC=14610.5
fit1
tsdisplay(residuals(fit1)) #显示残差

#残差白噪声检验
for(i in 1:2) print(Box.test(fit1$residual,lag=6*i))
#残差序列为白噪声，故模型拟合正确

# forecast
f.p1<-forecast(fit1,h=testsize,level=c(99.5))

print("mse is")
sum((f.p1[4]-testdata[2])^2)/testsize
#mes is 19.51638

fit2 <- Arima(tempseries, order=c(1,0,9),seasonal=list(order=c(0,1,0),period=365))
#结果：AIC=14589.92   AICc=14590.02   BIC=14654.7
fit2
tsdisplay(residuals(fit2)) #显示残差

#残差白噪声检验
for(i in 1:2) print(Box.test(fit2$residual,lag=6*i))
#残差序列为白噪声，故模型拟合正确


# forecast
f.p2<-forecast(fit2,h=testsize,level=c(99.5))

print("mse is")
sum((f.p2[4]-testdata[2])^2)/testsize
#mse is 19.49468

