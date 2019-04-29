source("clean.R")

################
# Lab 1 on YT
################

library(MASS)
#install.packages('ISLR')
library(ISLR)
names(Boston)
plot(medv~lstat, Boston)
fit1 = lm(medv~lstat, data = Boston)
fit11
summary(fit1)
abline(fit1, col ="red")
names(fit1)
confint(fit1)
# 5,10,15 are the new values that we want to predict and we ask for their confidence intervals
predict(fit1, data.frame(lstat = c(5,10,15)), interval = "confidence")
fit2 = lm(medv~lstat+age, data = Boston)
summary(fit2)
# all the variables are predictors except medv that is our target
fit3 = lm(medv~., data = Boston)
summary(fit3)
#age is no longer significant
#4 plots will be created and so we prepare the layout for this
par(mfrow = c(2,2)) 
plot(fit3)
#next command means redo the fit using the previous target (~) and removing age, indus
fit4 = update(fit3, ~. -age -indus); summary(fit4)
#Interactions: * means main effects and interaction
fit5= lm(medv~lstat+age, data = Boston); summary(fit5)
#Nonlinearity
fit6= lm(medv~lstat+I(lstat^2), data = Boston); summary(fit6)
attach(Boston) #<=> names in Boston are available in workspace

#we come back to previous layout for graphics
par(mfrow = c(1,1)) 
plot(medv~lstat, Boston)
#now we fit using fit6 to fit lstat and with a marker of size 20
points(lstat, fitted(fit6), col = "red", pch = 20)
#with the next line we try to fit with a polynom of lstat of degree 4
fit7=lm(medv~poly(lstat,4))
#to plot all the markers you can do 
plot(1:20,1:20,pch = 1:20, cex= 2)

#Now we change our dataset 
source("clean.R")
#fix open a dataframe with an editor to have a look at data
fix(Carseats)
names(Carseats)
summary(Carseats)
#summary in this gave will give also the number of 
#instances of each value of categorical variables
#we ask for a regression with all features + two Interactions.
fit1 = lm(Sales~.+Income:Advertising+Age:Price,Carseats)
#this command shows how R translate the categorical variable ShelveLoc in dummy ones
contrasts(Carseats$ShelveLoc) 
# Function in R

regplot = function(x,y)
{
  fit = lm(y~x)
  plot(x,y)
  abline(fit,col = "red")
}

attach(Carseats)
regplot(Price,Sales)

regplot2 = function(x,y,...)
{
  fit = lm(y~x)
  plot(x,y,...)
  abline(fit,col = "red")
}

regplot2(Price,Sales, xlab = "Price", ylab = "Sales", col = "blue", pch = 20)
