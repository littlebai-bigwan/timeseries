library(forecast)
library(tseries)
#1 load datas
trainsize = 144*0.9
testsize = 144-trainsize
data = read.csv("/Users/jessica/PycharmProjects/machinelearning/dataanalysis/data/passage.csv",nrows=trainsize)
testdata = read.csv("/Users/jessica/PycharmProjects/machinelearning/dataanalysis/data/passage.csv",skip=trainsize+1)
tempseries=ts(data[2],frequency = 12)
plot.ts(tempseries)
tsdisplay(tempseries,xlab="time",ylab="temperature",lag.max = 40)

#2 对时间序列进行拆分，得到三个组件：seasonal，trend，random，可以用加法模型处理季节性
tempcomp <- decompose(tempseries)
plot(tempcomp)

#3 平稳性检验：
tempcomp$random=na.omit(tempcomp$random)
adf.test(tempcomp$random)
# 通过Dickey-Fuller,可以看出序列random不平稳,因此holtwinters可能不一定适用


# 4 holtwinters模型
temppred <- HoltWinters(tempseries)
temppred[3:5]
# holtwinters 三alpha、beta和gamma 来分别对应当前点的水平、趋势部分和季节部分
# alpha=0.24394 ,beta=0.03450103,gamma=1

# 5 forecast
f.p1<-forecast(temppred,h=testsize,level=c(99.5))
plot(f.p1)
print("mse is")
sum((f.p1[4]-testdata[2])^2)/testsize
