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

#3 平稳性检验：
adf.test(tempseries)
# 通过Dickey-Fuller,可以看出序列平稳

tempcomp$random=na.omit(tempcomp$random)
adf.test(tempcomp$random)
# 通过Dickey-Fuller,可以看出序列random不平稳,因此holtwinters可能不一定适用


# holtwinters模型
temppred <- HoltWinters(tempseries)
temppred[3:5]
# alpha=0.5872089,beta=0,gamma=0.4939254

# forecast
f.p1<-forecast(temppred,h=testsize,level=c(99.5))
plot(f.p1)
print("mse is")
sum((f.p1[4]-testdata[2])^2)/testsize
