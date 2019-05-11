####################
# Lab 8 on YT Part 1
####################

source("clean.R")
require(ISLR)
require(tree)
attach(Carseats)
x11()
hist(Sales)
High = ifelse(Sales<=8, "No", "Yes")
Carseats = data.frame(Carseats, High)

tree.carseats = tree(High~.-Sales, data = Carseats)
summary(tree.carseats)
x11()
plot(tree.carseats)
text(tree.carseats, pretty = 0)
tree.carseats

set.seed(1011)
train = sample(1:nrow(Carseats), 250)
tree.carseats = tree(High~.-Sales, Carseats, subset = train)
x11()
plot(tree.carseats)
text(tree.carseats, pretty = 0)

tree.pred = predict(tree.carseats, Carseats[-train,], type = "class")
with(Carseats[-train,], table(tree.pred, High))
cv.carseats = cv.tree(tree.carseats, FUN = prune.misclass)
cv.carseats

x11()
plot(cv.carseats)
prune.carseats = prune.misclass(tree.carseats, best = 13)

x11()
plot(prune.carseats)
text(tree.carseats, pretty = 0)

tree.pred = predict(prune.carseats, Carseats[-train,], type = "class")
with(Carseats[-train,], table(tree.pred, High))

####################
# Lab 8 on YT Part 2
####################

require(randomForest)
require(MASS)
set.seed(101)
dim(Boston)
train = sample(1:nrow(Boston), 300)
?Boston

rf.boston = randomForest(medv~., data = Boston, subset = train)
rf.boston

oob.err = double(13)
test.err = double(13)

for(mtry in 1:13) {
  fit = randomForest(medv~., data = Boston, subset = train, mtry = mtry, ntree = 400)
  oob.err[mtry] = fit$mse[400]
  pred = predict(fit, Boston[-train,])
  test.err[mtry] = with(Boston[-train,], mean((medv-pred)^2))
  cat(mtry, " ")
}

x11()
matplot(1:mtry, cbind(test.err, oob.err), pch = 19,
        col = c("red", "blue"), type = "b", ylab = "Mean Squared Error")
legend("topright", legend = c("OOB", "TEST"),pch = 19,
       col = c("red", "blue"))

# Boosting

require(gbm)
boost.boston = gbm(medv~., data = Boston[train,], distribution = "gaussian", n.trees = 10000,
                   shrinkage = 0.01, interaction.depth = 4)

x11()
summary(boost.boston)

x11()
plot(boost.boston, i = "lstat")

x11()
plot(boost.boston, i = "rm")

n.trees = seq(from = 100, to = 10000, by = 100)
predmat = predict(boost.boston, newdata = Boston[-train,], n.trees = n.trees)
dim(predmat)
berr = with(Boston[-train,], apply((predmat-medv)^2, 2, mean))
plot(n.trees, berr, pch = 19, ylab = "Mean Squared Error",
     xlab = "# Trees", main = "Boosting Test Error")
abline(h=min(test.err), col = "red")
