#Code for loading correctly the file DataADJ.txt and to test scale

source("clean.R")
Data = read.table("DataADJ.txt",header=T)
nrows = dim(Data)[1]
ncols = dim(Data)[2]-1
X = Data[1:nrows,-2]
Y = Data$Temp
X_numeric = X[,-ncols]
X_scaled = scale(X_numeric)
X_unscaled = cbind(unscale(X_scaled, X_scaled),X[,ncols])