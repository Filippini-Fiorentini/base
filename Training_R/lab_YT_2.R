source("clean.R")

####################
# Lab 2 on YT Part 1
####################

require(ISLR) #like library
names(Smarket)
summary(Smarket)
?Smarket
#plot the data
pairs(Smarket, col = Smarket$Direction)
#Lagistic Regression
glm.fit = glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume, data = Smarket, family = binomial)
summary(glm.fit)
#Pvalues are high => it's due to the fact that variables may be strongly correlated
#we would need maybe to use some stuff like PCA here
glm.probs=predict(glm.fit, type = "response")
#the result is a vector of probability for the prediction
glm.probs[1:5]
glm.pred=ifelse(glm.probs, "Up", "Down")
#this turns probabilities into classification
attach(Smarket)
table(glm.pred, Direction)
#this is the classical table for prediction with n output labels.....
mean(glm.pred==Direction)
#we separate training and testing
train = Year<2005
glm.fit = glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,
              data = Smarket, family = binomial, subset = train)
#now it will work only on train to build the model
glm.probs=predict(glm.fit, newdata = Smarket[!train,], type = "response")
glm.pred=ifelse(glm.probs, "Up", "Down")
Direction.2005 = Smarket$Direction[!train]
table(glm.pred, Direction.2005)
mean(glm.pred==Direction.2005)
#worse than before => maybe overfitting => reduce number of features
glm.fit = glm(Direction~Lag1+Lag2,
              data = Smarket, family = binomial, subset = train)
#now it will work only on train to build the model
glm.probs=predict(glm.fit, newdata = Smarket[!train,], type = "response")
glm.pred=ifelse(glm.probs, "Up", "Down")
Direction.2005 = Smarket$Direction[!train]
table(glm.pred, Direction.2005)
mean(glm.pred==Direction.2005)

source("clean.R")

####################
# Lab 2 on YT Part 2
####################

require(ISLR) 
require(MASS) 

lda.fit = lda(Direction~Lag1+Lag2, data = Smarket, subset = Year < 2005)
lda.fit 
plot(lda.fit )
Smarket.2005 = subset(Smarket, Year == 2005)
#we test only on 2005 indeed subset do the subset of a dataframe => fast and easy 
lda.pred = predict(lda.fit, Smarket.2005)
lda.pred[1:5]
#it's not in a matrix format => error
class(lda.pred)
# now we know it's type: it's a list
data.frame(lda.pred)[1:5,]
table(lda.pred$class, Smarket.2005$Direction) 
#check prediction vs real
mean(lda.pred$class== Smarket.2005$Direction)
#how many did we get right

##K-Nearest Neighbors
#very simple but effective in many cases => do not throw it away
library(class)
?knn
attach(Smarket)
Xlag = cbind(Lag1, Lag2)
#we put in the same matrix those 2 indeces
train = Year < 2005
knn.pred = knn(Xlag[train,], Xlag[!train,], Direction[train], k = 1)
#this command take Xlag in the train part as training data and test on !train
#the response is Direction(of the training) and we want k = 1 neighbor classification
table(knn.pred, Direction[!train])
#also called confusion matrix
mean(knn.pred == Direction[!train])
# 0.5 <=> tossing a coin <=> 
