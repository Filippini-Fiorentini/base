source("linearRegression.R")
source("MSE.R")

Temp = read.table("temperatura.txt",header=T)

# columns: Temperatura, Sesso (cat), Freq_cardiaca

nrows = dim(Temp)[1]
ncols = dim(Temp)[2]

train_idx = 1:floor(nrows/2)
test_idx = floor(nrows/2):nrows

# drop categorical values
Temp_numeric = Temp[1:nrows,c(1,3)]

x_vec = Temp_numeric$Temperatura[train_idx]
y_vec = Temp_numeric$Freq_cardiaca[train_idx]

# weights
w_vec = rep(1,length(train_idx))

# linearRegression
theta_vec = linearRegression(x_vec,y_vec,w_vec)

x_line = seq(min(x_vec),max(x_vec),len=length(train_idx))
y_line = theta_vec[1] + theta_vec[2] %*% x_line

# plot
plot(x_vec,y_vec,col="red")
lines(x_line,y_line,col="blue")	# lines VS plot: append instead of creating new 
																# figure

# training error
#
# predicted values
train_pred = theta_vec[1] + theta_vec[2] %*% x_vec
#
train_err = MSE(train_pred,y_vec)