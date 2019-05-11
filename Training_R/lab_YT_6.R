Model Selection

In this way we should have the code well commented like with doxygen
or matlab live script
```{r}

####################
# Lab 6 on YT Part 1
####################

source("clean.R")
library(ISLR)
summary(Hitters)

```
There are some missing values 
```{r}
Hitters = na.omit(Hitters)
with(Hitters, sum(is.na(Salary)))
```
we eliminate NA and we check that there are no more NA left

BEST SUBSET REGRESSION
```{r}

library(leaps)
regfit.full = regsubsets(Salary~.,data = Hitters)
summary(regfit.full)
#subsets are not always nested
regfit.full = regsubsets(Salary~.,data = Hitters, nvmax = 19)
reg.summary = summary(regfit.full)
names(reg.summary)
plot(reg.summary$cp, xlab = "Number of variables", ylab = "Cp")
which.min(reg.summary$cp)

model with 10 sets is the best one 

x11()
plot(regfit.full, scale = "Cp")
coef(regfit.full, 10)
```

####################
# Lab 6 on YT Part 2
####################

#Forward Stepwise Selection

regfit.fwd = regsubsets(Salary~., data = Hitters, nvmax = 19, method = "forward")
summary(regfit.fwd)
x11()
plot(regfit.fwd, scale = "Cp")

# We use a validation set 
dim(Hitters)
set.seed(1)
train = sample(seq(263),180,replace = F)
train
regfit.fwd = regsubsets(Salary~., data = Hitters[train,], nvmax = 19, method = "forward")

val.errors = rep(NA,19)
x.test = model.matrix(Salary~., data = Hitters[-train,])

for (i in 1:19) {
  coefi = coef(regfit.fwd, id = i)
  pred = x.test[,names(coefi)]%*%coefi
  val.errors[i] = mean((Hitters$Salary[-train]-pred)^2)
}

plot(sqrt(val.errors), ylab = "Root MSE", ylim = c(300,400), pch = 19, type = "b")
points(sqrt(regfit.fwd$rss[-1]/180), col = "blue", pch = 19, type = "b")
legend("topright", legend = c("Training", "Validation"), col = c("blue", "black"))

# Function to predict values from models that don't have a prediction function

predict.regsubsets = function(object, newdata, id, ...)
{
  form = as.formula(object$call[[2]])
  mat = model.matrix(form, newdata)
  coefi = coef(object, id = id)
  mat[,names(coefi)]%*%coefi
}

####################
# Lab 6 on YT Part 3
####################

set.seed(11)
folds = sample(rep(1:10, length = nrow(Hitters)))
folds
table(folds)
cv.errors = matrix(NA,10,19)
for (k in 1:10) {
  best.fit = regsubsets(Salary~., data = Hitters[folds!=k,], nvmax = 19, method = "forward")
  for(i in 1:19) {
    pred = predict(best.fit, Hitters[folds==k,], id = i)
    cv.errors[k,i] = mean((Hitters$Salary[folds==k]-pred)^2)
  }
}
rmse.cv = sqrt(apply(cv.errors, 2, mean))
x11()
plot(rmse.cv, pch = 19, type = "b")

####################
# Lab 6 on YT Part 4
####################

library(glmnet)
x = model.matrix(Salary~.-1, data = Hitters)
y = Hitters$Salary

fit.ridge = glmnet(x,y, alpha = 0)
#alpha = 0 <=> Ridge
#alpha = 1 <=> Lasso
#alpha in (0,1) <=> Elastic Net
x11()
plot(fit.ridge, xvar = "lambda", label = T)
# we can see now the optimal value of the coefficients for different
# values of lambda
cv.ridge = cv.glmnet(x,y, alpha = 0)
x11()
plot(cv.ridge)
  
fit.lasso = glmnet(x,y)
x11()
plot(fit.lasso, xvar = "lambda", label = T)

cv.lasso = cv.glmnet(x,y)
x11()
plot(cv.lasso)
coef(cv.lasso)
# this gives us the coefficients of the best model

lasso.tr = glmnet(x[train,], y[train])
lasso.tr
pred = predict(lasso.tr, x[-train,])
dim(pred)
rmse = sqrt(apply((y[-train]-pred)^2, 2, mean))
plot(log(lasso.tr$lambda), rmse, type = "b", xlab = "Log(Lambda)")
lam.best = lasso.tr$lambda[order(rmse)[1]]
lam.best
coef(lasso.tr, s = lam.best)
