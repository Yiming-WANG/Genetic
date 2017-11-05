#install.packages("quantmod") 
library(quantmod)
library(timeSeries)

#download data
stockList <- c("AAPL","CSCO","DIS","HD","INTC","JPM","MMM","MSFT","NKE","XOM")
getSymbols(stockList, src = "yahoo")

#prepare historical stock data and charts
historyData <- cbind(AAPL[,6],CSCO[,6],DIS[,6],HD[,6],INTC[,6],JPM[,6],MMM[,6],MSFT[,6],NKE[,6],XOM[,6])
plot(as.timeSeries(historyData), at = "chic", col = .colorwheelPalette(10),lwd=2,xlab=c(1,2,3,4,5,6,7,8,9,0))
title("Historical Stock Price")

#prepare train data
returns.train <- lapply(stockList, function(s)
  monthlyReturn(eval(parse(text = s)),subset = "2010::2015"))
returns.train <- do.call(cbind,returns.train)
colnames(returns.train) <- stockList

#install.packages("timeSeries")
#plot stock returns of each month (train data)
library(timeSeries)
plot(as.timeSeries(returns.train), at = "chic", col = .colorwheelPalette(10),lwd=6,type="p")
title("Return Rate of Each Month")

n <- ncol(returns.train) 
R.train <- colMeans(returns.train) # average monthly returns
R.train
c.train <- cov(returns.train) # monthly returns covariance matrix 
s.train <- colSds(returns.train) # monthly returns Standard deviation
s.train

plot(s.train, R.train, type="n", panel.first = grid(),xlab = "Risk (Volatility of Monthly Returns)", ylab = "Average Monthly Returns")
text(s.train, R.train, names(R.train), col = .colorwheelPalette(10), font = 2,cex=1.5)


#prepare testing data
#random selection data
set.seed(1234)
x <- runif(10)
w.random <- x/sum(x)
w.random

w.even = c(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)
w.even

returns.t <- lapply(stockList, function(s)
  monthlyReturn(eval(parse(text = s)),subset = "2016"))
returns.test <- do.call(cbind,returns.t)
colnames(returns.test) <- stockList
R.test <- colMeans(returns.test) # average monthly returns
c.test <- cov(returns.test) # covariance matrix of monthly returns
s.test <- sqrt(diag(c.test)) # volatility of monthly returns

#install.packages("ggplot2")
library(ggplot2)
percent <- function(x, digits = 2, format = "f") {
  paste0(formatC(100 * x, format = format, digits = digits), "%")
}

pie(w.random, labels = percent(round(w.random,digit=4)), main ="Percentage of Each Stock By Random Selection",col = .colorwheelPalette(10),lty=0,cex =1.5 )
legend("topleft", stockList, cex = 1.5, fill = .colorwheelPalette(10),border = "white",box.col="white")

pie(w.even, labels = percent(round(w.even,digit=4)), main ="Percentage of Each Stock By equal weight",col = .colorwheelPalette(10),lty=0,cex =1.5)
legend("topleft", stockList, cex = 1.5, fill = .colorwheelPalette(10),border = "white",box.col="white")


#what we need is a high sharpe ratio portfolio. So, use the GA to search the highest sharpe ratio portfolio 
weights <- function(w) # normalised weights
{ drop(w/sum(w)) }

PortfolioReturn <- function(w,R) # Portfolio return
{ sum(weights(w)*R) }

sharpe <- function (w,R){
  (PortfolioReturn(w,R)-rfr)/sd(weights(w)*R)
}

p.risk <- function(w,R){
  sd(weights(w)*R)
}

#since we focus on monthly return, so we choose 1 month T-Bill yield today (May 28, 2017) as the risk free rate
rfr=0.00728

#We create a variable: risk adjustment. Because, we think the risk adjustment can help us to find 
#a portfolio with more stable historical performance, which might also bring more stable future performance (more predictable performance). 
#Since the average monthly return of the 10 stocks is 0.1, so, We choose 0.1^2*10% as a base line of risk.
#If the risk is more than 10%* 0.1^2 = 0.001, we reduce the sharpe ratio by risk*1000 
#If the risk is more than 20%* 0.1^2 = 0.002, we reduce the sharpe ratio again, by (risk-0.002)*1000 
#The more risk, the more punishment. So, if a portfolio has too much risk, unless its sharpe ratio is really high, it would not be chosen.
fitness <- function(w)
{
  riskAdjust=0
  if (p.risk(w,R.train)>0.001) riskAdjust=riskAdjust+(p.risk(w,R.train))*1000
  if (p.risk(w,R.train)>0.002) riskAdjust=riskAdjust+(p.risk(w,R.train)-0.002)*1000
  sharpe(w,R.train)- riskAdjust
}


#parameter tuning: 1. probability of mutation  2.probability of crossover  3.the generation

#Since we are trying to find the best solution of the fitness function (highest sharpe ratio). 
#It is not what we normally did: model building + predicting. 
#From the subset of the training data, it is impossible to find the best solution of fitness function for the whole training data set. 
#Which means cross-validation validation is not appropriate for parameter tuning. 
#It is better to write a function to do directly parameter tuning by comparing the GA@solution of different parameters.

#Before tuning the probability of mutation and probability of crossover, we need to think about the trade-off between run-time and accuracy.
#The more generation, the higher computation time. And of course, the result might be more close to best solution.
#We started by 4000 generation (maxiter = 4000), and default value for probability of mutation and probability of crossover

#install.packages("GA")
library("GA")
set.seed(1234)

GA.default <- ga(type = "real-valued", fitness = fitness,min = rep(0,n), max = rep(1,n), names = stockList,maxiter = 4000)
summary(GA.default)
plot(GA.default)

#we can see that, when generation is more than around 1000, the result of the algorithm only improved less than (6.8-6.5)/6.5 = 4.6%.
#But the computation time increased 4000/1000 -1 = 300%
#The result from 1000 generation, which is around 6.6 sharpe ratio (after risk adjustment) is already great enough. 
#Considering both the run time and accuracy, we would use 1000 as the value of maxiter(generation).


#parameter tuning: 1. probability of mutation  2.probability of crossover
#Mostly, researchers use crossover rate on level 0.7-0.9 and mutation on 0.05-0.2 ("Parameter tuning for configuring and analyzing evolutionary algorithms",A.E. Eiben, S.K. Smit1)
#So, we would like to search the best value in these 2 ranges.
pcrossover = 0.7#0.7 to 9;  increased 0.01 each time
pmutation = 0.05 #0.05 to 0.2; increased 0.01 each time
bestFit=0
pc=0
pm=0

while (pcrossover <=0.9)
{
  while (pmutation <=0.2){
    set.seed(1234)
    GA <- ga(type = "real-valued", fitness = fitness,min = rep(0,n), max = rep(1,n), names = stockList, pcrossover = pcrossover, pmutation =pmutation,maxiter = 1000)
    pmutation = pmutation + 0.01
    if (GA@fitnessValue > bestFit) {
      bestFit = GA@fitnessValue
      pc=pcrossover
      pm=pmutation
    }
  }
  pcrossover = pcrossover + 0.01
}

pc
pm

set.seed(1234)
GA <- ga(type = "real-valued", fitness = fitness,min = rep(0,n), max = rep(1,n), names = stockList,pcrossover = pc, pmutation =pm,maxiter = 1000)
summary(GA)
plot(GA)

w <- weights(GA@solution)
w

#plot weights of GA portfolio
pie(w, labels = percent(round(w,digit=4)), main ="Percentage of Each Stock By GA Selection",col = .colorwheelPalette(10),lty=0, cex = 1.5)
legend("topleft", stockList, cex = 1.5, fill = .colorwheelPalette(10),border = "white",box.col="white")

#portfolio performance: Sharpe and risk
performances<- function(R){
  port.per.tr = cbind(sharpe(w.even,R),sharpe(w.random,R),sharpe(w,R))
  colnames(port.per.tr) <- c("even.Sharpe","random.Sharpe","GA.Sharpe")
  print(port.per.tr)
  
  port.rsk.tr = cbind(p.risk(w.even,R),p.risk(w.random,R),p.risk(w,R))
  colnames(port.rsk.tr) <- c("even.rsk","random.rsk","GA.rsk")
  print(port.rsk.tr)
}
performances(R.train)
performances(R.test)


#Plot 3 portfolio return
daily2016 <- lapply(stockList, function(s)
  dailyReturn(eval(parse(text = s)),subset = "2016"))
daily2016 <- do.call(cbind,daily2016)
colnames(daily2016) <- stockList

daily2016$port=1
daily2016$port.random=1
daily2016$port.even=1
daily2016[,1:10]=daily2016[,1:10]+1

for(i in 2:nrow(daily2016)){
  daily2016$port[i]=daily2016$port[i-1]*sum(w*as.numeric(daily2016[i-1,1:10]))
  daily2016$port.random[i]=daily2016$port.random[i-1]*sum(w.random*as.numeric(daily2016[i-1,1:10]))
  daily2016$port.even[i]=daily2016$port.even[i-1]*sum(w.even*as.numeric(daily2016[i-1,1:10]))
}

plot(as.timeSeries(daily2016[,11:13]), plot.type="single",at = "chic", col = .colorwheelPalette(4),lwd=3)
title("Portfolio Return")
legend("topleft", c("GA","Random","Equal Weight"), cex = 2.5, fill = .colorwheelPalette(4),border = "white",bg= "white")
