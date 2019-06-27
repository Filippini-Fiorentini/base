#####################################################################
#							LIBRARIES							    #
####################################################################

library(MASS)	# for LDA & QDA

#####################################################################
#						INITIALIZE CONTAINERS					    #
####################################################################

# vector
v1 = c(2,3,4,5)				# [2,3,4,5]
v2 = rep(10,3)				# [10, 10, 10]
v3 = seq(2,6,len=4)			# linspace(2,6,4)
v4 = seq(2,6,by=0.1)		# 2:0.1:6
v5 = 2:6					# 2:6
v6 = vector("double",10)	# std::vector<double> v6(10)

# list
l1 = vector("list",10)		# se questo non è immorale... -.-"

# named list (struct)
exam1 <- list (
			  course = 'Applied Statistics',  
              date = '27/09/2014',
              enrolled = 7,
              corrected = 6,
              student_id = as.character(c(45020,45679,46789,43126,
							              42345,47568,45674)),
              evaluation = c(30,29,30,NA,25,26,27)
              )
              
# dataframe
exam2 <- data.frame(
					student_id = factor(as.character(c(45020,45679,
													   46789,43126,
													   42345,47568,
													   45674))),
  					evaluation_W = c(30,29,30,NA,25,26,17), 
					evaluation_O = c(28,30,30,NA,28,27,NA), 
					evaluation_P = c(30,30,30,30,28,28,28),
					outcome  = factor(c('Passed','Passed','Passed',
										'To be repeated','Passed',
										'Passed','To be repeated'))
					)

# matrix
m1 = rbind(c(11,13,15),c(12,14,16))
m2 = cbind(c(11,12),c(13,14),c(15,16))
m3 = matrix(data = c(1,2,3,4,5,6), nrow = 2, ncol = 3, byrow = F)
m4 = matrix(0,3,4) 	# zeros(3,4)

# NOTE: all operations are by default componentwise (use %*% otherwise)

# inversion of matrices (and solution of linear systems)
A = matrix(c(11,13,12,14), ncol=2, nrow=2, byrow=TRUE)
b = c(1,1)
invA = solve(A)
x = solve(A,b)



#####################################################################
#								JITTERING							#
####################################################################

# Jittering - this may be useful if we think that our data are
# collected with small error, iot check whether the results are
# robust against small variations
# To be precise, to do JITTERING means to add to them a small 
# random (gaussian...or according to any distribution we like) value, 
# iot shake them
set.seed(280787)
dataset = dataset + cbind(rnorm(dim(dataset)[[1]], sd=0.025))



#####################################################################
#						CATEGORICAL VARIABLES						#
####################################################################

# sample of categorical variables (from vector)
district = c('MI','MI','VA','BG','LO','LO','CR','Alt','CR','MI','Alt',
			  'CR','LO','VA','MI','Alt','LO','MI')
district 0 factor(district,levels=c('MI','LO','BG','CR','VA','Alt'))

# table of absolute frequences
resass = table(district)

# table of relative frequences
resrel = table(district)/length(district)


#####################################################################
#							  IMPORT DATA							#
####################################################################

dataset = read.table('filename_in.txt', header=T)
attach(dataset)

n = dim(dataset)[[1]]
p = dim(dataset)[[2]]

rownm = dimnames(dataset)[[1]]		# or rownames(dataset)
colnm = dimnames(dataset)[[2]]		# or colnames(dataset)

head(dataset)	# to visualize only the first rows

otherdata = load('filename_in_2.RData')



#####################################################################
#				   QUANTITATIVE DATA (UNIVARIATE)					#
####################################################################

colMeans(dataset)			# mean of each feature (mean by columns)
rowMeans(dataset)			# mean of each unit (mean by rows)

# sapply(obj,func) applies func to obj
sapply(dataset, mean)
sapply(dataset, sd)
sapply(dataset, var)

cov(dataset)				# covariance matrix of the dataset
cor(dataset)				# correlation matrix of the dataset

# Attention: rounded zeros!
round(sapply(dataset,mean),1)
round(cov(dataset),1)
round(cor(dataset),1)

### verify Gaussian assumption on one feature
#
# 	Shapiro-Wilk test:	to verify (quantitatively) the Gaussian  
# 						assumption on the distribution generating 
#						sample
shapiro.test(col1)

#	qqplot:				to verify (qualitatively) the Gaussian 
#						assumption on the distribution generating 
#						sample
x11()
{
	qqnorm(col1) 				# quantile-quantile plot
	qqline(col1, col='red') 	# theoretical line
}

# having X (nx2) and wanting to test the two columns one at a time
x11()
{
	par(mfrow=c(2,2))
	
	hist(X[,1], prob=T, ylab='density', xlab='X.1', 
		 main='Histogram of X.1',ylim=c(0,0.45))
	lines((-1000):1000 /100, dnorm((-1000):1000 /100,mean(X[,1]),
	 	  sd(X[,1])), col='blue', lty=2)
	
	hist(X[,2], prob=T, ylab='density', xlab='X.2', 
		 main='Histogram of X.2',ylim=c(0,0.45))
	lines((-1000):1000 /100, dnorm((-1000):1000 /100,mean(X[,2]),
		 sd(X[,2])), col='blue', lty=2)

	qqnorm(X[,1], main='QQplot of X.1',xlab='theoretical quantiles', 
		   ylab='sample quantiles')
	qqline(X[,1])

	qqnorm(X[,2], main='QQplot of X.2',xlab='theoretical quantiles', 
		   ylab='sample quantiles')
	qqline(X[,2])
}

# Univariate Box-Cox transformation
# 		We compute the optimal lambda of the univariate Box-Cox 
#		transformation
# 		(command powerTransform())
lambda.x = powerTransform(x) 
# 		lambda<1: observations <1 are "spread", observations >1 are 
#		"shrinked"

# 		Transformed sample with the optimal lambda (command bcPower())
bc.x = bcPower(x, lambda.x$lambda)


### t-test for the mean value of the quantity 
# H0: mean==m0 vs H1: mean!=m0
#
#	manually:
alpha = .05
sampleMean = mean(col1)
m0 = 11.5
sampleSd = sd(col1)
n = length(col1)
tstat = (sampleMean - m0) / (sampleSd/sqrt(n))
cfrT = qt(1 - alpha/2, n-1)
abs(tstat) < cfrT  # cannot reject H0 (accept H0)

# pvalue
pval = ifelse(tstat >= 0, (1 - pt(tstat, n-1))*2, pt(tstat, n-1)*2)

# confidence interval
CI = c(
		inf = sampleMean - sampleSd/sqrt(n) * qt( 1 - alpha/2, n-1 ), 
       	center = sampleMean, 
       	sup = sampleMean + sampleSd/sqrt(n) * qt( 1 - alpha/2, n-1 )
      )


#	automatically:
t.test(col1, mu = m0, alternative = 'two.sided', confLevel = 1-alpha)


### simple linear regression
# Fit of the linear model (command lm)
# Model: yy = beta0 * xx + eps, eps~N(0,sigma^2)
regression <- lm(yy ~ xx)

coef(regression)
vcov(regression)
residuals(regression)
fitted(regression)

# plot results of regression
x11()
{
	plot(xx, yy, asp=1, cex=0.75)
	abline(coef(regression))
	points(xx, fitted(regression), col='red', pch=19)
	legend('bottomright',c('Obs.','Fit','Reg. line'),
			col=c('black','red','black'),lwd=c(1,1,1),
			lty=c(-1,-1,1),pch=c(c(1,19,-1)))
	title(main='Linear regression (yy vs xx)')
}

# test F "by hand" (H0: beta0=0 vs H1: beta0!=0)
SSreg = sum((fitted(regression) - mean(yy))^2)
SSres = sum(residuals(regression)^2)
SStot = sum((yy - mean(yy))^2)
n = length(yy)
Fstat = (SSreg/1) / (SSres/(n-2))
P = 1 - pf(Fstat, 1, n-2)

# confidence and prediction intervals (command predict)
newdata = data.frame(xx=c(10,11,12))
pred_nd = predict(regression, newdata)

CI_nd = predict(regression, newdata, interval = 'confidence', 
				level = .99)
PI_nd = predict(regression, newdata, interval = 'prediction', 
				level = .99)

# plot confidence and prediction intervals
x11()
{
	plot(xx, yy, asp=1, ylim=c(18.5,27.5), cex=0.5)
	abline(coef(regression))
	points(xx, fitted(regression), col='red', pch=20)
	points(newdata, pred_nd, col='blue', pch=16)
	matlines(rbind(newdata, newdata), t(PI_nd[,-1]),
			 type="l", lty=2, col='dark grey', lwd=2)
	matpoints(rbind(newdata, newdata), t(PI_nd[,-1]), pch="-",
			  lty=2, col='dark grey', lwd=2, cex=1.5)
	matlines(rbind(newdata, newdata), t(CI_nd[,-1]), type="l",
			 lty=1, col='black', lwd=2)
	matpoints(rbind(newdata, newdata), t(CI_nd[,-1]), pch="-",
			  lty=1, col='black', lwd=2, cex=1.5)
	legend('bottomright', c('Obs.','Fit','Reg. line','Pred. new',
							'CI','PI'),
			col=c('black','red','black','blue','black','dark grey'),
			lwd=c(1,1,1,1,2,2),lty=c(-1,-1,1,-1,1,2),
			pch=c(c(1,19,-1,19,-1,-1)))
	title(main='Linear regression (yy vs xx)')
}

# diagnostic of residuals
x11()
{
	par (mfrow=c(2,2))
	boxplot(residuals(regression), main='Boxplot of residuals')
	qqnorm(residuals(regression))
	plot(xx, residuals(regression), main='Residuals vs xx')
	abline(h=0, lwd=2)
	plot(fitted(regression), residuals(regression), 
		 main='Residuals vs fitted yy')
	abline(h=0, lwd=2)
}



#####################################################################
#				   VISUALIZATION of MULTIVARIATE DATA				#
####################################################################

# customize color according to categorical label
labelIdx = 4
labelCol = factor(dataset[,labelIdx])
colorLabel = ifelse(labelCol == '1', 'red', 'blue')

# Scatterplot
#
#	each pannel represents a cloud of data; the coordinates of each plot
#	represent two variables. Each point is a statistical unit (an 
#	observation), while the axes represent two variables
#	We see the shape of the data --> we can determine whether there 
#	is a correlation (and which kind of correlation) between the 
#	variables
#
x11()
{
	pairs(dataset, col=colorLabel, pch=16)
}

# Boxplot
#
#	it is a box whose middle line is the median of the data; the 
#	lower line is the first quartile (it leaves 25% of the observation 
#	of the left), while the upper bound is the third quartile. Points 
#	outside the "baffi" are outliers
#	--> it gives information about the symmetry of the distribution
#
x11()
{
	boxplot(dataset, col='gold')
}

# when points are in very different ranges, we can typically use log 
# in order to make them comparable
x11()
{
	boxplot(log(dataset), col='gold')
}

# we plot the data centered wrt the means
x11()
{
	boxplot(scale(x=dataset, center=T, scale=F), las=2, col='gold')
}

# Stratified boxplots (same scale)
#
x11()
{
	par(mfrow = c(1,4))	
	boxplot(dataset$col1 ~ labelCol, 
			col=c('red','blue'), main='col1', ylim=range(dataset))
	boxplot(dataset$col2 ~ labelCol, 
			col=c('red','blue'), main='col2', ylim=range(dataset))
	boxplot(dataset$col3 ~ labelCol, 
			col=c('red','blue'), main='col3', ylim=range(dataset))
	boxplot(dataset$col4 ~ labelCol, 
			col=c('red','blue'), main='col4', ylim=range(dataset))
}

# Histogram
#
#	VERY IMPORTANT NOTE
#	put two histograms one next to the other only when they have the
#	same scale, otherwise comparisons are not possible (actually, 
#	the real problem is that people make them in any case, but they 
#	are wrong!)
x11()
{
	layout(cbind(c(1,1), c(2,3)), widths=c(2,1), heights=c(1,1))
	plot(col1, col2, asp=1, col=colorLabel, pch=16)
	hist(col1, prob=T, xlim=c(-10,15))
	hist(col2, prob=T, xlim=c(-10,15))
}


#####################################################################
#				   VISUALIZATION of CATEGORICAL DATA				#
#####################################################################

# Pie chart (no ordering of levels)
x11()
{
	pie(table(district),col=rainbow(length(levels(district))))
}

# Barplot (levels are ordered)
#
#	NOTE: a barplot is not an histogram! Hist are for numerical data,
# 	while barplots are for categorical data.
#	Barplot: height proportional to frequency
#	Hist: area proportional to frequency
#
x11()
{
	barplot(table(district)/length(district))
}



#####################################################################
#				   PRINCIPAL COMPONENT ANALYSIS					    #
#####################################################################

# we standardize the dataset (if neeeded)
dataset <- scale(dataset)
dataset <- data.frame(dataset)

PC = princomp(dataset, scores=T)
summary(PC)

# standard deviation of the components (i.e. sqrt of the eigenvalues)
PC$sd
# proportion of variance explained by each PC
PC$sd^2 / sum(PC$sd^2)
# cumulative proportion of explained variance
cumsum(PC$sd^2) / sum(PC$sd^2)

# loadings (recall: coefficients of the linear combination of the 
# original variables that defines each principal component)
#
# they are the directions of the principal components (i.e. the
# eigenvectors)
PCloads = PC$loadings

# graphical representation of the loadings of the first 
# principal components (the ones that explain 80% of variance)
var_explained = 0.8
nmaxPC = min(which(cumsum(PC$sd^2)/sum(PC$sd^2) > var_explained))
x11()
{
	par(mar = c(1,4,0,2), mfrow = c(nmaxPC,1))
	for(i in 1:nmaxPC) 
		barplot(PCloads[,i], ylim = c(-1, 1))
}

# scores
PCscores = PC$scores
PCscores = data.frame(PCscores)

# graphical representation of the scores
#
# a concentration in this plot shows that some transformation on 
# the data could be clever. For example, there is some boundary 
# effect, maybe due to the fact that data are positive
x11()
{
	plot(PCscores[,1:2])
	abline(h=0, v=0, lty=2, col='grey')
}

# variability of the original variables / scores
x11()
{
	layout(matrix(c(1,2),2))
	boxplot(dataset, las=2, col='gold', main='Original variables')
	boxplot(PCscores, las=2, col='gold', main='Principal components')
}


# plot of explained variance
x11()
{
	layout(matrix(c(2,3,1,3),2,byrow=T))
	plot(PC, las=2, main='Principal components', ylim=c(0,4.5e7))
	barplot(sapply(dataset,sd)^2, las=2, main='Original Variables', 
			ylim=c(0,4.5e7), ylab='Variances')
	plot(cumsum(PC$sd^2)/sum(PC$sd^2), type='b', 
	 	 axes=F, xlab='number of components', 
     	 ylab='contribution to the total variance', ylim=c(0,1))
	abline(h=1, col='blue')
	abline(h=var_explained, lty=2, col='blue')
	box()
	axis(2,at=0:10/10,labels=0:10/10)
	axis(1,at=1:ncol(dataset),labels=1:ncol(dataset),las=2)
}

# color according to... SEE LAB3

# Projection on the space generated by the k-th principal component
meanF = colMeans(dataset)
for (i in 1:8)
{
	projection = matrix(meanF, dim(dataset)[[1]], dim(dataset)[[2]],
  					    byrow=T) + PCscores[,i] %*% t(PCloads[,i])
}
x11(width=21, height=7)
{
	par(mfrow=c(2,5))
	matplot(t(dataset), type='l', main = 'Data', ylim=range(dataset))
	matplot(meanF, type='l', main = '0 PC', lwd=2, ylim=range(dataset))
	for(i in 1:8)
	{
  		matplot(t(projection), type='l', main = paste(i, 'PC'),
  		  		ylim=range(dataset))
		matplot(meanF, type='l', lwd=2, add=T)
	}
}

# Projection on the space generated by the first k principal components
projection = matrix(meanF, dim(dataset)[[1]], dim(dataset)[[2]],
					byrow=T)
for (i in 1:8)
{
	projection = projection + PCscores[,i] %*% t(PCloads[,i])
}
x11(width=21, height=7)
{
	par(mfrow=c(2,5))
	matplot(t(dataset), type='l', main = 'Data', ylim=range(dataset))
	matplot(meanF, type='l', main = 'First 0 PCs', lwd=2, 
			ylim=range(dataset))
	for(i in 1:8)
	{
  		matplot(t(projection), type='l', 
  				main = paste('First', i, 'PCs'),
	  		    ylim = range(dataset))
		matplot(meanF, type='l', lwd=2, add=T)
	}
}



#####################################################################
#					TESTING FOR MULTIVARIATE NORMALITY			    #
#####################################################################

### H0: X_i distributed as Gaussian
### H1: X_i not distributed as Gaussian

### Approach 1: look at some linear combinations of the original 
#				variables (example: single components, scores of the 
#				principal components)
# 				Indeed: gaussian variable --> all the linear 
#				combinations are gaussian
#				Of course, this is really a proof only if we find a 
# 				counterexample: we cannot really check ALL the linear 
#				combinations
# SEE UNIVARIATE ANALYSIS

### Approach 2:	consider the Mahalanobis distances of the data from 
#				the (sample) mean and test if they are a sample from 
#				a chi-square distribution
# 				
#				Recall:
# 				Theorem: if X~N(mu,Sigma) r.v. in R^p, det(Sigma)>0
#          		then d2(X,mu)=(X-mu)'Sigma^-1(X-mu) ~ Chi-sq(p)

### Approach 3:	mcshapiro.test()
load('../../mcshapiro.test.RData')
mcshapiro.test(dataset)

### Multivariate Box-Cox transformation
lambda = powerTransform(cbind(x,y))
BC.x = bcPower(x, lambda$lambda[[1]])
BC.y = bcPower(y, lambda$lambda[[2]])



#####################################################################
#							REMOVE OUTLIERS			 			    #
#####################################################################

# based on Mahalanobis distance
d2 = matrix(mahalanobis(dataset, colMeans(dataset), cov(dataset)))
upperthr = 7.5

x11()
{
	plot(d2, pch=ifelse(d2 < upperthr, 1, 19))
}
x11()
{
	plot(dataset, pch=ifelse(d2 < upperthr, 1, 19))
}

dataset_noout = dataset[which(d2<upperthr),]



#####################################################################
#						PROBABILITY REGION		 				   	 #
#####################################################################

# Let X=(X1 X2 X3)'~N(mu, Sigma) a Gaussian random vector with
# mu=(1 1 1)' and Sigma=cbind(c(5,3,1),c(3,5,1),c(1,1,1)).
# a) Identify a region A such that P((X1 X2)' \in A)=0.9
# b) Identify a region A2 such that P((X1 X2)' \in A2 | X3=1)=0.9
# c) Having reported in a plot the graphs of the two regions, order in 
#    increasing order the following probabilities:
#    P((X1 X2)' \in A )
#    P((X1 X2)' \in A2)
#    P((X1 X2)' \in A  | X3=1)
#    P((X1 X2)' \in A2 | X3=1)

library(car)
mu=c(1,1,1)
Sigma=cbind(c(5,3,1),c(3,5,1),c(1,1,1))

### a) Consider only (X1 X2)'
#
# If the 3x1 vector is gaussian, its component (X1 X2)' is gaussian
eigen(Sigma[1:2,1:2])
#
# Direction of the axes:
eigen(Sigma[1:2,1:2])$vectors
#
# Center:
M = mu[1:2]
#
# Radius of the ellipse:
r = sqrt(qchisq(0.9,2))
#
# Length of the semi-axes:
r*sqrt(eigen(Sigma[1:2,1:2])$values)


### b) Consider the conditional distribution (X1 X2)'|X3=1
#
# Functions to compute the mean and the covariance matrix of the 
# conditional distribution
muCond <- function(mu1,mu2,Sig11,Sig12,Sig22,x2)
{
	return(mu1+Sig12%*%solve(Sig22)%*%(x2-mu2))
}
#
SigCond <- function(Sig11,Sig12,Sig22)
{
	Sig21 = t(Sig12)
	return(Sig11 - Sig12 %*% solve(Sig22) %*% Sig21)
}
# Center of the ellipse
Mc = muCond(mu1=mu[1:2], mu2=mu[3], Sig11=Sigma[1:2,1:2],
			Sig12=Sigma[1:2,3], Sig22=Sigma[3,3], x2=1)
Sigmac = SigCond(Sig11=Sigma[1:2,1:2],Sig12=Sigma[1:2,3],
				 Sig22=Sigma[3,3])
#
# Direction of the axes
eigen(Sigmac[1:2,1:2])$vectors
#
# Radius of the ellipse
r = sqrt(qchisq(0.9,2))
#
# Length of the semi-axes
r*sqrt(eigen(Sigmac[1:2,1:2])$values)


### Plot of (a) and (b)
x11()
{
	plot(M[1], M[2], xlim=c(-10,15), ylim=c(-10,15), col='blue', 
	 	 pch=19, xlab='X.1', ylab='X.2', asp=1)
	ellipse(center=M, shape=cbind(Sigma[1:2,1:2]), radius=r, 
			col='blue')
	ellipse(center=as.vector(Mc), shape=Sigmac, radius=r, col = 'red')
	abline(h=0, v=0, lty=2, col='grey')
	abline(a=0,b=1,col='grey',lty=2)
	abline(a=2,b=-1,col='grey',lty=2)
}



######################################################################
#						 HYPOTHESIS TESTING						     #
#####################################################################

# Premiss: general rule to perform a test
# 1)  Formulate the test (and test the Gaussian assumption, if needed)
# 2)  Compute the test statistics (our case: T^2 Hotelling)
#		T^2 = n(\bar{X} - \mu_0)^T S^{-1} (\bar{X} - \mu_0)
#		~
#		\frac{(n-1)p}{n-p} F(p, n-p)
# 3a) Having set the level of the test, verify whether the test  
#     statistics belongs to the region of rejection (i.e., if there is   
#     statistical evidence to reject H0)
# 3b) Compute the p-value of the test

# Be careful to apply transitivity in the statistical context!!
# {A not significantly (stat.) different from B} [A = B]   
#
#                       and
# {B not significantly (stat.) different from C} [B = C]   
#
#                  does NOT imply
# {A not significantly (stat.) different from C} [A = C]   
#
# Note. If we don't reject H0, we are not proving that A=B but
#       we are saying that we can't prove that A!=B


### TEST1:
#			H0: mu=mu0 vs H1: mu!=mu0
#			with mu0=c(1,0)
#
alpha = 0.01	# level of the test
m0 = c(1,0)	# target value for the mean
#
# 1) test the Gaussian assumption
mcshapiro.test(dataset)
#
n = dim(dataset)[[1]]
p = dim(dataset)[[2]]
R = range(dataset)
sampleMean = colMeans(dataset)
sampleCov = cov(dataset)
invCov = solve(sampleCov)
#
# 2) T2 Statistics
T2 = n * (sampleMean - m0) %*% invCov %*% (sampleMean - m0)
#
# 3a) Radius of the ellipsoid
cfrFisher = ((n-1)*p/(n-p)) * qf(1-alpha,p,n-p)	# qf: quantile of
												# a Fisher
#
# Test: 
T2 < cfrFisher	# Rejection region: {T2 > cfrFisher}
				# (we reject for large values of the T2 statistics)
#
# 3b) Compute the p-value 
Pvalue = 1 - pf(T2*(n-p)/((n-1)*p), p, n-p)

# 4)  Characterize the region:	compute the centre, direction of 
#     							the principal axes, length of the axes
#
# centre: mean
sampleMean
#
# Directions of the principal axes:		
#		eigenvectors of the covariance matrix 
#		(or of (covariance matrix / n), which is the same)
eigen(sampleCov/n)$vectors
#
# Length of the semi-axes of the ellipse:	
# 		sqrt(radius)*sqrt(eigenvalues of (covariance matrix / n))
r = sqrt(cfrFisher)
r * sqrt(eigen(sampleCov/n)$values) 
# Warning: Conf Reg => x.cov/n 

# Plot test result (rejection region VS confidence region)
x11()
{
	plot(dataset, asp = 1)
	# region of rejection (centred in m0!)
	ellipse(m0, shape=sampleCov/n, sqrt(cfrFisher), col = 'blue', 
			lty = 2, center.pch = 4, center.cex=1.5, lwd=2)
	# center: m0
	points(m0[1], m0[2], pch = 4, cex = 1.5, lwd = 2, col ='blue')
	# sample mean
	points(sampleMean[1], sampleMean[2], pch = 4, cex = 1.5, 
		   lwd = 2, col ='red')
	# confidence region
	# {m \in R^2 s.t. n * (x.mean-m)' %*% (x.cov)^-1 %*% (x.mean-m) < 
	#  cfr.fisher }
	ellipse(sampleMean, sampleCov/n, sqrt(cfrFisher), col = 'red', 
			lty = 2, lwd=2, center.cex=1)
}

# Remark: the radius and the shape of the ellipse are the same, but 
# the centre changes:
# - Rejection region: the centre is the mean mu0 under H0 (blue ellipse)
# - Confidence region: the centre is the sample mean (red ellipse)

# Which relation between the two ellipses?
# - If the rejection region does NOT contain the sample mean (i.e., we
#   are in the acceptance region), then we cannot reject H0 
#   (i.e., if the sample mean falls within the ellipse we accept H0)
# - If the mean under H0 (mu0) is contained in the confidence region
#   of level 1-alpha, then we do not reject H0 at level alpha

# => the confidence region of level 1-alpha contains all the mu0
#    that we would accept at level alpha

# Note: by definition, the confidence region of level 1-alpha
# produces ellipsoidal regions that contain the true mean
# 100(1-alpha)% of the times. If H0 is true (i.e., mu0 is 
# the true mean), those ellipsoidal regions will contain mu0 
# 100(1-alpha)% of the times

### TEST2:
#			Asymptotic test on the mean --> \chi^2 test
#			H0: mu==m0 vs H1: mu!=m0
#			with m0=c(1,0)
#
#	T^2 = n(\bar{X} - \m_0)^T S^{-1} (\bar{X} - \m_0) ~ \chi^2(p)
#
# Note 1: the higher the number of data, the better the approximation!
# Note 2: we don't need to verify the Gaussian assumption!

T2A = n * (sampleMean - m0) %*%  invCov  %*% (sampleMean - m0)
cfrChisq = qchisq(1-alpha, p)
T2A < cfrChisq

PvalueA = 1 - pchisq(T2A, p)

# Comparison between exact and asymptotic
x11(width=14, height=7)
{
	# Rejection regions
	par(mfrow=c(1,2))
	plot(dataset, asp = 1,main='Comparison rejection regions')
	ellipse(m0, shape=sampleCov/n, sqrt(cfrFisher), col = 'blue', 
			lty = 1, center.pch = 4, center.cex=1.5, lwd=2)
	points(m0[1], m0[2], pch = 4, cex = 1.5, lwd = 2, col ='blue')
	ellipse(m0, sampleCov/n, sqrt(cfrChisq), col = 'lightblue', 
			lty = 1, center.pch = 4, center.cex=1.5, lwd=2)
	points(m0[1], m0[2], pch = 4, cex = 1.5, lwd = 2, 
		   col ='lightblue')
	legend('topleft', c('Exact', 'Asymptotic'),
			col=c('blue','lightblue'), lty=c(1), lwd=2)

	# Confidence regions
	plot(dataset, asp = 1,main='Comparison of confidence regions')
	points(sampleMean[1], sampleMean[2], pch = 4, cex = 1.5, 
		   lwd = 2, col ='red')
	ellipse(sampleMean, sampleCov/n, sqrt(cfrFisher), col = 'red', 
			lty =1, lwd=2, center.cex=1)
	points(sampleMean[1], sampleMean[2], pch = 4, cex = 1.5, 
		   lwd = 2, col ='orange')
	ellipse(sampleMean, sampleCov/n, sqrt(cfrChisq), col = 'orange', 
			lty = 1, center.pch = 4, center.cex=1.5, lwd=2)
	legend('topleft', c('Exact', 'Asymptotic'),
			col=c('red','orange'), lty=c(1), lwd=2)
}


# Simultaneous T2 confidence intervals on the coordinate directions
# Recall: these are projections of the ellipsoidal confidence region
SCI_T2 = data.frame(
			cbind(
				inf = sampleMean - sqrt(cfrFisher*diag(sampleCov)/n),
	           	center = sampleMean, 
            	sup = sampleMean + sqrt(cfrFisher*diag(sampleCov)/n)
            )
         )
            	
# Bonferroni intervals on the coordinate directions
k = p # number of intervals I want to compute (set in advance)
cfrT = qt(1 - alpha/(2*k), n-1)
Bf = data.frame(
		cbind(
			inf = sampleMean - cfrT*sqrt(diag(sampleCov)/n),
            center = sampleMean, 
            sup = sampleMean + cfrT*sqrt(diag(sampleCov)/n)
        )
     )
     
# Note: Here we have three confidence regions:
# 		(1) ellipse, (2) T2 rectangle, (3) Bonferroni rectangle
# In general, univariate confidence intervals for k given directions 
# are associated with polygonal regions with 2*k sides
# In this example, for k=2 the Bonferroni rectangle is smaller than the
# T2 rectangle. For increasing values of k, the Bonferroni region will 
# become larger and larger (cf. multiple testing)


# Plot confidence intervals
x11()
{
	par(mfrow=c(1,1))
	plot(dataset, asp = 1,main='Confidence and rejection regions')

	# rejection region
	ellipse(m0, shape=sampleCov/n, sqrt(cfrFisher), col = 'blue', 
			lty = 2, center.pch = 16, center.cex=1.5, lwd=2)
	points(m0[1], m0[2], pch = 4, cex = 1.5, lwd = 2, col ='blue')

	# confidence region	
	ellipse(sampleMean, shape=sampleCov/n, sqrt(cfrFisher), 
			col = 'red', lty = 2, center.pch = 16, 
			center.cex=1.5, lwd=2)
	points(sampleMean[1], sampleMean[2], pch=19, cex=1.25, col='red')

	# simultaneous T2 intervals
	rect(SCI_T2$inf[[1]], SCI_T2$inf[[2]], 
		 SCI_T2$sup[[1]], SCI_T2$sup[[2]], 
		 border='red', lwd=2)
		 
	# Bonferroni intervals
	rect(Bf$inf[[1]], Bf$inf[[2]], 
		 Bf$sup[[1]], Bf$sup[[2]], 
		 border='orange', lwd=2)
	
	legend('topleft', 
		   c('Rej. Reg.', 'Conf. Reg','T2-sim', 'Bonferroni'),
		   col=c('blue','red','red','orange'),
		   lty=c(2,2,1,1), lwd=2)

}

# Plot of the confidence intervals in each direction of interest 
# (with global coverage alpha)
x11()
{
	nn = dim(SCI_T2)[[1]]
	matplot(1:nn, 1:nn, pch='', ylim=R, xlab='Variables',
			ylab='Confidence intervals along a component', 
        	main='Confidence intervals')
	
	for(i in 1:nn)
	{
		# Simultaneous T2 intervals
		segments(i, SCI_T2[i,1], i, SCI_T2[i,3], lwd=2, col=i, lty=3)
		# Bonferroni
		segments(i, Bf[i,1], i, Bf[i,3], lwd=2, col=i)
	}
	# inf
	points(1:nn, SCI_T2[,1], pch='-', col=1:nn)		# T2
	points(1:nn, Bf[,1], pch='-', col=1:nn)			# Bf
	# sup
	points(1:nn, SCI_T2[,3], pch='-', col=1:nn)		# T2
	points(1:nn, Bf[,3], pch='-', col=1:nn)			# Bf
	# center
	points(1:nn, SCI_T2[,2], pch='x', col=1:nn)		# T2
	points(1:nn, Bf[,2], pch=16, col=1:4)			# Bf

	# projection of m0 along the given directions
	points(1:nn, m0, lwd=3, col='orange')
}

# Simulation to compute level & power of a test --> LAB6


# Determine a direction along which we reject H0
# 		Recall: we reject the global H0 if in at least one direction 
#				we observe a 'high' value of the statistics t^2 
#				(univariate), i.e., we reject the global H0 if we 
#				reject the univariate test at least in direction 
#				(max(t^2))
# 		-->
# 		Worst direction: direction along which the t^2 statistics 
#						 (univariate) is maximized

# => From the theory
#    - the maximum is realized (Hotelling T2-statistics)
T2
#    - the distribution of the maximum is known (and is?)
#    - the direction along which the maximum is realized is known
worst = invCov %*% (sampleMean - m0)
worst = worst / sqrt(sum(worst^2))
# Angle with the x-axis:
thetaWorst = atan(worst[2]/worst[1]) + pi

# Confidence interval along the worst direction:
CIworst = data.frame(
			cbind(
				inf = sampleMean %*% worst - 
					  sqrt(cfrFisher*(t(worst)%*%invCov%*%worst)/n),
	           	center = sampleMean %*% worst,
            	sup = sampleMean %*% worst + 
            		  sqrt(cfrFisher*(t(worst)%*%invCov%*%worst)/n)
            )
         )
m0%*%worst
(CIworst[[1]] < m0%*%worst) & (m0%*%worst < CIworst[[2]])   
# Reject H0: a'mu = a'delta.0 in direction a = worst

m1 = worst[2]/worst[1] # worst direction

# Extremes of CIworst in the coordinate system (x,y):
xMin = CIworst[[1]]*worst
xMax = CIworst[[3]]*worst
m1_ort = -worst[1] / worst[2]
q_min_ort = xMin[2] - m1_ort*xMin[1]
q_max_ort = xMax[2] - m1_ort*xMax[1]

x11()
{
	plot(dataset, asp=1, pch=1, main='Worst direction',ylim=c(-15,60))
	abline(h=m0[1], v=m0[2], col='grey35')
	ellipse(center=sampleMean, shape=sampleCov/n, 
			radius=sqrt(cfrFisher), lwd=2)
	points(m0[1], m0[2], pch=16, col='grey35')
	rect(xleft = SCI_T2[1,1], ybottom = SCI_T2[2,1], 
		 xright = SCI_T2[1,3], ytop = SCI_T2[2,3], 
		 border='red', lwd=1,lty=2)

	abline(0, m1, col='forestgreen', lty=1,lwd=2)
	abline(q_min_ort, m1_ort, col='forestgreen', lty=2,lwd=1)
	abline(q_max_ort, m1_ort, col='forestgreen', lty=2,lwd=1)
}


# Bonferroni intervals on other directions
k = 3
theta = c(0, pi/4, pi/2)
Bfintervals = vector("list",length(theta))
ST2intervals = vector("list",length(theta))

for(i in seq_along(theta))
{
	# direction
	a = c(cos(theta[i]), sin(theta[i]))
	a_orth = c(-sin(theta[i]), cos(theta[i]))

	# Bonferroni	
	Bfintervals[[i]] = data.frame(
						cbind(
							inf = sampleMean - 
								  sqrt(var(as.matrix(dataframe) %*% 
								  					 a) / n ) * 
  								  qt(1 - alpha/(2*k), n-1) * a,
							center = sampleMean,
							sup = sampleMean + 
								  sqrt(var(as.matrix(dataframe) %*% 
								  					 a) / n ) * 
  								  qt(1 - alpha/(2*k), n-1) * a
						 )
					  )
	
	# Simultaneous T2
	ST2intervals[[i]] = data.frame(
						cbind(
							inf = sampleMean - 
								  sqrt(var(as.matrix(dataframe) %*% 
								  					 a) / n ) * 
  								  sqrt(cfr.fisher) * a,
							center = sampleMean,
							sup = sampleMean + 
								  sqrt(var(as.matrix(dataframe) %*% 
								  					 a) / n ) * 
  								  sqrt(cfr.fisher) * a
						 )
					  )	
}

x11()
{
	plot(dataset, asp=1, main=paste0('Confidence regions (k=',
									 k,')'))
	ellipse(center=sampleMean, shape=sampleCov/n, 
			radius=sqrt(cfrFisher), lwd=1)
	abline(h=0, v=0, col='grey')
	points(m0[1], m0[2], pch=16, col='grey35', cex=1.25)
	for(i in seq_along(theta))
	{
		# direction
		a = c(cos(theta[i]), sin(theta[i]))
		a_orth = c(-sin(theta[i]), cos(theta[i]))
	
		lines(rbind(Bfintervals[[i]]$sup + 100*a_orth,
               		Bfintervals[[i]]$sup - 100*a_orth), 
	          col='orange', lty=1,lwd=1)
         
		lines(rbind(Bfintervals[[i]]$inf + 100*a_orth,
               		Bfintervals[[i]]$inf - 100*a_orth), 
         	  col='orange', lty=1,lwd=1)
         	  
		lines(rbind(ST2intervals[[i]]$sup + 100*a_orth,
               		ST2intervals[[i]]$sup - 100*a_orth), 
	          col='red', lty=1,lwd=1)
         
		lines(rbind(ST2intervals[[i]]$inf + 100*a_orth,
               		ST2intervals[[i]]$inf - 100*a_orth), 
         	  col='red', lty=1,lwd=1)	
	}
}


#####################################################################
#						PAIRED OBSERVATIONS						    #
####################################################################

# we compute the sample of differences
D = data.frame(DIFF1 = dataset[,1] - dataset[,3], 
			   DIFF2 = dataset[,2] - dataset[,4])

### T2 Hotelling Test 
# H0: delta==delta.0 vs H1: delta!=delta.0
# with delta.0=0
#
# PROCEED AS IN HYPOTHESIS TESTING



#####################################################################
#						 REPEATED MEASURES							#
####################################################################

# Test: H0: C%*%mu=0 vs H1: C%*%mu!=0

n = dim(dataset)[[1]]
p = dim(dataset)[[2]]
q = 4 						# to be set!

S = cov(dataset)
M = sapply(dataset,mean)

# contrast matrix
C = matrix(0, q-1, q)

# contrast with baseline
C[,1] = -1
for(i in 1:(q-1))
	C[i,i+1] = 1
	
# day-by-day contrast
for(i in 1:(q-1))
{
	C[i,i] = 1
	C[i,i+1] = -1
}

alpha = .05
delta0 = rep(0, q-1)

Md = C %*% M 
Sd = C %*% S %*% t(C)
Sdinv = solve(Sd)

T2 = n * t( Md - delta0 ) %*% Sdinv %*% ( Md - delta0 )

cfrFisher = ((q-1)*(n-1)/(n-(q-1))) * qf(1-alpha, (q-1), n-(q-1)) 

T2 < cfrFisher 		# reject H0 if FALSE

Pvalue = 1 - pf(T2 * (n-(q-1))/((q-1)*(n-1)), (q-1), n-(q-1))

# confidence intervals on the components
k = q - 1   # number of increments (i.e., dim(C)[1])
cfrT = qt(1-alpha/(2*k),n-1)

Bf = data.frame(
		cbind(
			inf = Md - cfrT*sqrt(diag(Sd)/n), 
			center = Md, 
			sup = Md + cfrT*sqrt(diag(Sd)/n)
		 )
	  )

SCI_T2 = data.frame( 
			cbind( 
				inf = Md - sqrt(cfrFisher*diag(Sd)/n) , 
				center = Md, 
				sup = Md + sqrt(cfrFisher*diag(Sd)/n)
			)
		  )

# Note: if we change contrast matrix, the result of the test is the
# same, but the confidence intervals change

### Example:
#			"the drug decreases the pressure of two units with 
#			respect to the baseline at both 8 and 16 hs, and its 
#			effect vanishes in 24 hs from the drug administration"
C = matrix(c(-1, 1, 0, 0,
             -1, 0, 1, 0,
             -1, 0, 0, 1), 3, 4, byrow=T)
m0 = c(-2,-2,0)



#####################################################################
#			TEST FOR MEAN OF TWO INDEPENDENT GAUSSIAN			   #
###################################################################

# Test H0: mu1=mu2  vs  H1: mu1!=mu2
# i.e.,
# Test H0: mu1-mu2=c(0,0)  vs  H1: mu1-mu2!=c(0,0)

n1 = dim(dataset1)[1]
n2 = dim(dataset2)[1]
p  = dim(dataset1)[2]

# sample mean, covariance matrices and matrix Spooled
mean1 = sapply(dataset1,mean)
mean2 = sapply(dataset2,mean)
cov1  = cov(dataset1)
cov2  = cov(dataset2)
Sp    = ((n1-1)*cov1 + (n2-1)*cov2)/(n1+n2-2)

# we compare the matrices
list(S1=cov1, S2=cov2, Spooled=Sp)

alpha   = .01
delta0  = c(0,0)
Spinv   = solve(Sp)

T2 = n1*n2/(n1+n2) * 
	 (mean1-mean2-delta0) %*% Spinv %*% (mean1-mean2-delta0)

cfrFisher = (p*(n1+n2-2)/(n1+n2-1-p)) * qf(1-alpha,p,n1+n2-1-p)
T2 = cfrFisher 

Pvalue = 1 - pf(T2/(p*(n1+n2-2)/(n1+n2-1-p)), p, n1+n2-1-p)

SCI_T2 = data.frame( 
			cbind( 
				inf = mean1 - mean2 - 
					  sqrt(cfrFisher*diag(Sp)*(1/n1+1/n2)),
				center = mean1 - mean2, 
				sup = mean1 - mean2 + 
					  sqrt(cfrFisher*diag(Sp)*(1/n1+1/n2))
			)
		  )



#####################################################################
#							ONE-WAY ANOVA						   #
###################################################################

# one variable observed over g levels
# -->
# treatCol: name of the column reporting the level
# varCol:	name of the column storing the value of the variable

attach(dataset)

n = dim(dataset)[[1]]		# total number of observations
ng = table(treatCol)    		# number of observations in each group
treat = levels(treatCol)    # levels of the treatment
g = length(treat)     		# number of levels (i.e., of groups)

### verify the assumptions:
Ps = vector("double", g)
Var = vector("double", g)
for (i in seq_along(Ps))
{
	# 1) normality (univariate) in each group
	Ps[[i]] = shapiro.test(varCol[treatCol==treat[i]])$p
	# 2) same covariance structure (= same sigma^2)
	Var[[i]] = var(varCol[treatCol==treat[i]])
}

# test of homogeneity of variances
# H0: sigma.1 = sigma.2 = sigma.3 = sigma.4 = sigma.5 = sigma.6 
# H1: there exist i,j s.t. sigma.i!=sigma.j
bartlett.test(varCol, treatCol)

# Model: weigth.ij = mu + tau.i + eps.ij; eps.ij~N(0,sigma^2)
# Test:
#		H0: tau.1 = tau.2 = tau.3 = tau.4 = tau.5 = tau.6 = 0
#		H1: (H0)^c
# i.e.,
#		H0: The treatments don't have effect
#	    (= "observations belong to a single population")
#		H1: At least one treatment has effect
#		(= "observations belong to 2, 3, 4, ... populations")

fit = aov(varCol ~ treatCol)
summary(fit)

### How to read the summary:
#             	  Df   Sum Sq      Mean Sq      F value     Pr(>F)    
#  treat		(g-1) SStreat  SStreat/(g-1)  Fstatistic    p-value 
#  Residuals  	(n-g) SSres     SSres/(n-g)    
#
# note: pvalues are the ones of test [H0: tau.i=0 for every i]                
#---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1 


# If the test is effective, which supplement is responsible for 
# this? To see this, we need to do g*(g-1)/2 comparisons.
# We use Bonferroni
k = g*(g-1)/2
alpha = 0.05

Mean = mean(varCol)
Meang = tapply(varCol, treatCol, mean)

SSres = sum(residuals(fit)^2)

S = SSres/(n-g)

# CI for all the differences
CIrangeBonf = vector("list", k)		# with Bonferroni correction
CIrange = vector("list", k)			# without Bonferroni correction
h = 1
for(i in 1:(g-1)) 
{
	for(j in (i+1):g) 
	{
    		print(paste(treat[i],"-",treat[j]))
	    	diff = Meang[i]-Meang[j]
    		quantileB = qt(1-alpha/(2*k), n-g)
    		quantile = qt(1-alpha/2, n-g)
	    	nng = (1/ng[i] + 1/ng[j])
    	
    		intervBonf = as.numeric(c(diff - quantileB * sqrt( S * nng),
        	           		      	 diff + quantileB * sqrt( S * nng)))
        
        	interv = as.numeric(c(diff - quantile * sqrt( S * nng),
        	           		  diff + quantile * sqrt( S * nng)))
    	
    	CIrangeBonf[[h]] = intervBonf
    		CIrange[[h]] = interv
    		h = h + 1
	}
}

# plot CI for all the differences
x11(width = 14, height = 7)
{
	par(mfrow=c(1,3))
	plot(treatCol, varCol, xlab='treat', col = rainbow(6), las=2)

	# with Bonferroni correction
	h = 1
	plot(c(1,g*(g-1)/2), range(CIrangeBonf), pch='', 
		 xlab='pairs treat', ylab='Conf. Int. tau (Bonferroni)')
	for(i in 1:(g-1)) 
	{
		for(j in (i+1):g) 
		{
		    lines(c(h,h), CIrangeBonf[[h]], col='grey55')
		    points(h, Mediag[i]-Mediag[j], pch=16, col='grey55')
    		points(h, CIrangeBonf[[h]][1], col=rainbow(6)[j], 
    				pch=16)
		    points(h, CIrangeBonf[[h]][2], col=rainbow(6)[i], 
		    			pch=16)
		    h = h+1
		}
	}
	abline(h=0)

	# without Bonferroni correction
	h = 1
	plot(c(1,g*(g-1)/2), range(CIrange), pch='', xlab='pairs treat', 
		 ylab='Conf. Int. tau')
	for(i in 1:(g-1)) 
	{
		for(j in (i+1):g) 
		{
		    lines(c(h,h), CIrange[[h]], col='grey55')
		    points(h, Mediag[i]-Mediag[j], pch=16, col='grey55')
    		points(h, CIrange[[h]][1], col=rainbow(6)[j], pch=16)
		    points(h, CIrange[[h]][2], col=rainbow(6)[i], pch=16)
		    h = h+1
		}
	}
	abline(h=0)
}

# matrix of pvalues for all the differences
Pvalues_matr = matrix(0, g, g)
for(i in 1:g) 
{
	for(j in i:g)
	{
	    Pvalues_matr[i,j] = (1-pt(abs((Mediag[i]-Mediag[j]) / 
    						sqrt(S * (1/ng[i] + 1/ng[j]))), n-g))*2}
	for(j in 1:i)
	{
	    Pvalues_matr[i,j] = (1-pt(abs((Mediag[i]-Mediag[j]) / 
    						sqrt(S * (1/ng[i] + 1/ng[j]))), n-g))*2}
	Pvalues_matr[i,i] = 0
}

# vector of p-values (since the matrix above is symmetric, being
# the test between 1 and 3 the same as the test between 3 and 1,
# of course)
Pvalues = NULL
for (i in 1:(g-1))
	Pvalues = c(Pvalues, Pvalues_matr[i, (i+1):g])

# Bonferroni correction
Pvalues_bonf = p.adjust(Pvalues, 'bonf')

# Correction according to the false discovery rate 
# (Benjamini-Hockberg)
Pvalues_fdr = p.adjust(Pvalues, 'fdr')

# plot of pvalues
x11()
{
	plot(1:k, Pvalues, ylim=c(0,1), type='b', pch=16, 
		 col='grey55', xlab='pairs treat', main='P-values')
	abline(h=alpha, lty=2)
	lines(1:k, Pvalues_bonf, col='blue', pch=16, type='b')
	lines(1:k, Pvalues_fdr, col='red', pch=16, type='b')
	legend('topleft', c('Not corr.', 'Bonf.', 'BH'), 
			col=c('grey55', 'blue', 'red'), pch=16)

}

detach(dataset)



#####################################################################
#							ONE-WAY MANOVA						   #
###################################################################

# g > 1, p > 1

attach(dataset)		# suppose that the column named groups stores
					# a categorical variable that identifies the
					# different groups among which variables are
					# subdivided ('group1', 'group2', ... up to
					# 'group_g')

# Identify different levels
groupNames = c('group1','group2','group3')
groupCol = factor(groups, labels=groupNames)

g = length(groupNames)		# number of groups

Levels = vector("list",g)
nns = vector("integer",g)	# number of observations in each group
for (i in 1:g)
{
	Levels[[i]] = which(groupCol==groupNames[[i]])
	nns[[i]] = length(Levels[[i]])
}
n = sum(nns)				# total number of observations

# remove from the dataset the categorical column used to identify
# groups
dataset = dataset[, -grep('groups',colnames(dataset))]

p = dim(dataset)[[2]]

# To check the assumptions we need to perform testing
# --> elliptical clouds for gaussian data
# --> elliptical clouds with almost the same orientation if the
#	  covariance structure is the same for all the goups
# if one of these does not occur, we have problems with 
# the assumption!

# --> data exploration
colours = rep(rainbow(g), each = 50)
x11()
{
	pairs(dataset, col=colours, pch=16)
}

Ps = vector("double",g)
Ss = vector("list",g)
ms = vector("list",g)
S = cov(dataset)			# estimates Sigma
m = sapply(dataset,mean)	# estimates mu
Ssg = NULL
mmg = NULL
for(i in 1:g)
{
	current_ds = data.frame(dataset[Levels[[i]],])
	
	# 1)  normality (multivariate) in each group (g tests)
	Ps[[i]] = mcshapiro.test(current_ds)$p
	
	# 2) same covariance structure (= same covariance matrix Sigma)
	Ss[[i]] = cov(current_ds)
	Ssg = rbind(Ssg, Ss[[i]])
	
	# visto che siamo qui...
	
	# estimates mu.i = mu + tau.i
	ms[[i]] = sapply(current_ds,mean)
	mmg = rbind(mmg, ms[[i]])	
}

# Qualitatively:
x11(width=21)
{
	par(mfrow=c(1,g))
	for(i in 1:g)
	{
		image(Ss[[i]], col=heat.colors(100), 
			  main=paste0('Cov. S',i), asp=1, axes = FALSE, 
			  breaks = quantile(Ssg, (0:100)/100, na.rm=TRUE))
	}
}

# Note: We can verify the assumptions a posteriori on the 
# residuals of the estimated model 

# Model: X.ij = mu + tau.i + eps.ij; 
# where eps.ij~N_p(0,Sigma), X.ij, mu, tau.i in R^4

# Test:
#		H0: tau.1 = tau.2 = tau.3  = (0,0,0)'
#		H1: (H0)^c
# that is
#		H0: The membership to an iris species hasn't any 
#			significant effect on the mean of X.ij (in any 
#			direction of R^4) 
#		H1: There exists at least one direction in R^4 along 
#			which at least two species have some feature 
#			significantly different

fit = manova(as.matrix(dataset) ~ groups)
summary.manova(fit, test="Wilks")   # Exact tests for p<=2 or g<=3 
									# already implemented in R
# Note: since g=3 the test is exact

# If the treatment was effective, who's the responsible for this?

# Via ANOVA: for each of the p=4 variables we perform an ANOVA test
#            to verify if the membership to a group has influence
#            on the mean of the variable (we explore separately the
#            4 axes directions in R^4)
summary.aov(fit)

# Each of the 4 variables is significantly influenced by the  
# factor species
# Note. This analysis does NOT say: 
#       a) which group differ
#       b) with respect to which variables the groups in (a) differ
# => As for the ANOVA, we build confidence intervals (many more!)

# Via Bonferroni
alpha = 0.05
k = p*g*(g-1)/2
qT = qt(1-alpha/(2*k), n-g)

W = summary.manova(fit)$SS$Residuals

CIs = vector("list",g)
h = 1
for(i in 1:(g-1)) 
{
	for(j in (i+1):g) 
	{
		diff = ms[[i]] - ms[[j]]
		fact = sqrt(diag(W)/(n-g) * (1/nns[i] + 1/nns[j]))

		CIs[[h]] = data.frame(
						cbind(
							inf = diff - qT * fact,
							center = diff,
							sup = diff + qT * fact
						)
					)
		h = h + 1
	}
}

# Print confidence intervals
x11()
{	
	par(mfrow=c(1,4), las=2)
	for(i in 1:p)
	{
		plot(c(1,g*(g-1)/2), ylim=c(-4,4), xlim=c(1,3), pch='', 
		     xlab='pairs treat', ylab=paste0('CI tau',i), 
       		 main=paste0('CI tau_',colnames(dataset)[i]))
       	for(j in 1:g)
		{
			lines (c(j,j), 
			       c(CIs[[j]]$inf[[i]],CIs[[j]]$sup[[i]]))
       		points(j, CIs[[j]]$center[[i]], pch=16)
       		points(j, CIs[[j]]$inf[[i]], col=rainbow(g)[j], pch=16)
       		points(j, CIs[[j]]$sup[[i]], col=rainbow(g)[j], pch=16)
       		abline(h=0)
       	}
	}
}



#####################################################################
#							TWO-WAYS ANOVA						   #
###################################################################

# g1 > 1, g2 > 2, p = 1

g1 = length(levels(factor1))
g2 = length(levels(factor2))

M = mean(xx)
Mfactor1 = tapply(xx, factor1, mean)
Mfactor2 = tapply(xx, factor2, mean)
Minter = tapply(xx, factors1_2, mean)

# Model with interaction (complete model): 
# X.ijk = mu + tau.i + beta.j + gamma.ij + eps.ijk; 
# where 
#		eps.ijk~N(0,sigma^2), 
#     	i=1:g1 (effect of 1st factor), j=1:g2 (effect of 2nd factor)

fit_aov2_int = aov(xx ~ factor1 + factor2 + factor1:factor2)
summary.aov(fit_aov2_int)

# Tests:
# 1) 	H0: gamma11 = gamma.12 = gamma.21 = gamma.22 = 0    vs   
#		H1: (H0)^c
# i.e.,
#		H0: There is no significant interaction between the 
#			two factors in terms of performances
#		H1: There exists a significant interaction between the 
#			two factors in terms of performances

# --> if H0, we can try to remove interactions

# Additive model:
# X.ijk = mu + tau.i + beta.j + eps.ijk; 
# where
#		eps.ijk~N(0,sigma^2), 
#     	i=1:g1 (effect of 1st factor), j=1:g2 (effect of 2nd factor)

fit_aov2_ad = aov(xx ~ factor1 + factor2)
summary.aov(fit_aov2_ad)

# Remark: by removing the interaction, the residual degrees of 
# freedom increase! 

# Tests:
# 2)	H0: tau.1 = tau.2 = 0
#		H1: (H0)^c
# i.e.,
#		H0: The effect of 1st factor doesn't significantly influence 
#			performances 
#		H1: The effect of 1st factor significantly influences 
#			performances

# --> if H0, we can try to remove factor1

# Reduced additive model (ANOVA one-way, b=2): 
# X.jk = mu + beta.j + eps.jk;
# where
#		eps.ijk~N(0,sigma^2), 
#     	j=1:g2 (effect of 2nd factor)

fit_aov1 = aov(xx ~ factor2)
summary.aov(fit_aov1)

# Tests:
# 3)	H0: beta.1 = beta.2 = 0
#		H1: (H0)^c
# i.e.,
#	    H0: The effect of 2nd factor doesn't significantly influence 
#			performances
#		H1: The effect of 2nd factor significantly influences 
#			performances

# ETC, FINISCI DI GUARDARE LAB08



#####################################################################
#							  	LDA								   #
###################################################################

### 2 classes, univariate

attach(dataset)

A = which(group=='A')	# Group A: favorable clinical outcome
B = which(group=='B')	# Group B: unfavorable clinical outcome

x11()
{
	plot(dataset[,1], dataset[,2], pch=19, 
		 col=c(rep('blue',8),rep('red',5)),
	 	 xlab=colnames(dataset)[[1]], ylab=colnames(dataset)[[2]])
}

### Idea: we aim to find a "rule" to classify patiens as Group A
### or B, given the measurments of col1 and col2
### In fact, we only consider the variable col1 since there isn't  
### statistical evidence to state that there is a difference in mean 
### along the component col2 
#
# actually, "statistical evidence" means, in this case, that from
# the plot we do not see an evidence
#
### -> Exercise: build a confidence region of level 95% for the 
###    difference of the means between the two group (independent 
###    populations) and verify the previous statement


# LDA (univariate) 
#
# Assumptions:
# 1) if L=i, X.i ~ N(mu.i, sigma.i^2), i=A,B: GAUSSIANITY IN EVERY
#											  GROUP
# 2) sigma.A=sigma.B:	SAME VARIANCE
# 3) c(A|B)=c(B|A):		EQUAL MISSCLASSIFICATION COSTS

# NOTE: R does not allow to explicitly impose a cost of 
# missclassification

# verify assumptions 1) e 2):
# 1) normality (univariate) within the groups
shapiro.test(dataset[A,1])
shapiro.test(dataset[B,1])

# 2) equal variance (univariate)
bartlett.test(col1 ~ group)

# what if we have to reject the assumptions? We cannot use LDA,
# but we can try, for example, if the variances are not the same,
# with QDA
# If we miss gaussianity...either we know the distribution and
# in that case we can use a Bayes classifier, or we must choose
# another one

# dimension of the dataframe
nA = length(A)
nB = length(B)
n  = nA + nB

# Prior probabilities: in this case, we estimate the priors from
# the samples
PA = nA / n		# proportion of units in group A
PB = nB / n		# proportion of units in group B

library(MASS)
LDA = lda(group ~ col1)		# we want to explain the label (group)
							# through the value of the features
							# (in this case, only col1...in other
							# cases, feature1 + feature2 + ...)
# Note: if we don't specify the prior, they are estimated
# from the sample

# posterior probability and classification for new patient we 
# want to classify)
newdata = data.frame(col1 = seq(-10, 35, 0.5))
# The command predict() returns a list containing (see the help)
# - the class associated with the highest probability (i.e. the 
#	class in which we classify the new patient)
predict(LDA, newdata)$class
# - the posterior for the two classes
LDA_A = predict(LDA, newdata)$posterior[,1]
LDA_B = predict(LDA, newdata)$posterior[,2]
# - in lda: the coordinates of the canonical analysis of Fisher
predict(LDA, newdata)$x

x11()
{
	plot(col1[A], rep(0, length(A)), pch=16, col='blue', ylim=c(0,1),
    	 xlab='x', ylab='estimated posterior', main="LDA", 
	     xlim=range(col1))
	points(col1[B], rep(0, length(B)), pch=16, col='red')
	abline(v=0, col='grey')
	lines(newdata[,1], LDA_A, type='l', col='blue', xlab='x',
		  ylab='estimated posterior', main="LDA")
	points(newdata[,1], LDA_B, type='l', col='red')
	abline(h = 0.5)
	legend(-10, 0.9, legend=c('P(A|X=x)', 'P(B|X=x)'),
		   fill=c('blue','red'), cex = 0.7)
}

# set prior probabilities instead of estimating them from the samples
LDA1 = lda(group ~ col1, prior=c(0.05,0.95))

# etc as before



### 3 classes, bivariate

# verify assumptions --> see one-way MANOVA

attach(dataset)		# suppose that the column named groups stores
					# a categorical variable that identifies the
					# different groups among which variables are
					# subdivided ('group1', 'group2', ... up to
					# 'group_g')

# Identify different levels
groupNames = c('group1','group2','group3')
groupCol = factor(groups, labels=groupNames)

g = length(groupNames)		# number of groups

Levels = vector("list",g)
nns = vector("integer",g)	# number of observations in each group
for (i in 1:g)
{
	Levels[[i]] = which(groupCol==groupNames[[i]])
	nns[[i]] = length(Levels[[i]])
}
n = sum(nns)				# total number of observations

# remove from the dataset the categorical column used to identify
# groups
dataset = dataset[, -grep('groups',colnames(dataset))]

p = dim(dataset)[[2]]

Ss = vector("list",g)
ms = vector("list",g)
S = cov(dataset)			# estimates Sigma
m = sapply(dataset,mean)	# estimates mu
Sp = matrix(0,p,p)
for(i in 1:g)
{
	current_ds = data.frame(dataset[Levels[[i]],])

	Ss[[i]] = cov(current_ds)
	ms[[i]] = sapply(current_ds,mean)
	
	Sp = Sp + (nns[[i]] - 1) * Ss[[i]]
}
Sp = Sp / (n - g)

# Linear Discriminant Analysis (LDA)
LDA = lda(dataset, groupCol)

# "coefficients of linear discriminants" and "proportion of trace":
# Fisher discriminant analysis. 
# In particular:
# - coefficients of linear discriminants: versors of the canonical 
#   directions [to be read column-wise]
# - proportion of trace: proportion of variance explained by the 
#   corresponding canonical direction
names(LDA)

LDA_pred = predict(LDA, dataset)
names(LDA_pred)

# Compute the APER
LDA_pred$class   # assigned classes
groupCol	     # true labels
table(class_true=groupCol, class_assigned=LDA_pred$class)

errors = (LDA_pred$class != groupCol)	# TRUE if the assigned
										# class is different
										# from the true class

sum(errors)
length(groupCol)

APER = sum(errors)/length(groupCol)

# Remark: this is correct only if we estimate the prior with the   
#         empirical frequences! Otherwise:
#prior = c(1/3,1/3,1/3)
#g = 3
#misc = table(class_true=groupCol,
#			  class_assigned=LDA_pred$class)
#APER = 0
#for(i in 1:g)
#	APER = APER + sum(misc[i,-i])/sum(misc[i,]) * prior[i]

# Compute the estimate of the AER by cross-validation 
LdaCV = lda(dataset, groupCol, 
			CV=TRUE)  #CV=T: use leave-one-out cross-validation

LdaCV$class
groupCol
table(class_true=groupCol, class_assigned_CV=LdaCV$class)

errorsCV = (LdaCV$class != groupCol)
errorsCV
sum(errorsCV)

AERCV = sum(errorsCV)/length(groupCol)

# Remark: correct only if we estimate the priors through the 
# sample frequencies!

# Plot the partition induced by LDA
x11()
{
	plot(dataset, main='Title', xlab='xlab',
		 ylab='ylab', pch=20)
	for(i in 1:g)
	{
		points(dataset[Levels[[i]],], col=rainbow(i), pch=20)	
	}
	legend(min(dataset[,1]), max(dataset[,2]), 
		   legend=levels(groupCol),
		   fill=c('red','green','blue'), cex=.7)
	points(LDA$means, pch=4,col=c('red','green','blue') , 
		   lwd=2, cex=1.5)
}





#####################################################################
#							  KNN								   #
###################################################################

# It does not require any assumption, but for the fact that we must
# have a metrics

# Let's consider a non-parametric classifier
library(class)

KNN = knn(train = col1, 
		  test = newdata, 
		  cl = group, 		# cl:	label
		  k = 3, 			# k:	number of neighbors to be 
							# 		considered
		  prob=T) 			# prob=T: the classification is 
							#		  made 
							# 		  according to the
							#		  probabilities

KNN_class = (KNN == 'B')+0 
KNN_B = ifelse(KNN_class == 1, attributes(KNN)$prob, 
               1 - attributes(cito.knn)$prob)

x11()
{
	plot(newdata[,1], LDA_B, type='l', col='red', lty=2, xlab='x',
		 ylab='estimated posterior', 
		 main="propability of belonging to class RED")
	points(newdata[,1], KNN_B, type='l', col='red', lty=4)
	abline(h = 0.5)
	points(col1[A], rep(0, length(A)), pch=16, col='blue')
	points(col1[B], rep(0, length(B)), pch=16, col='red')
	legend(-10, 0.75, legend=c('LDA','knn'), lty=c(2,4), col='red')
}

# let's change k
x11(width = 28, height = 21)
par(mfrow=c(3,4))
{
	for(k in 1:12)
	{
	  KNN = knn(train = col1, test = newdata, cl = group, k = k, 
	  			prob=T)
	  KNN_class = (KNN == 'B')+0 
	  KNN_B = ifelse(KNN_class==1, attributes(KNN)$prob, 
	  				 1 - attributes(KNN)$prob)
  
	  plot(newdata[,1], LDA_B, type='l', col='red', lty=2, xlab='x',
  		   ylab='estimated posterior', main=k)
	  points(newdata[,1], KNN_B, type='l', col='black', lty=1, lwd=2)
	  abline(h = 0.5)
	}
}

detach(dataset)




#####################################################################
#							  QDA								   #
###################################################################

# Gaussian assumption + different covariances ---> QDA

QDA = qda(dataset, groupCol)		# REMEMBER: by specifying 
								# only data and label, the
								# priors are set 
								# automatically

QDA_pred = predict(QDA, dataset)

# compute the APER
QDA_pred$class
groupCol
table(class_true=groupCol, class_allocated=QDA_pred$class)

errorsq = (QDA_pred$class != groupCol)
errorsq

APERq = sum(errorsq)/length(groupCol)
APERq
# Remark: correct only if we estimate the priors through the 
# sample frequencies!

# Compute the estimate of the AER by cross-validation 
QdaCV = qda(dataset, groupCol, CV=T)
QdaCV$class
groupCol
table(class_true=groupCol, class_allocated=QdaCV$class)

errorsqCV = (QdaCV$class != groupCol)
errorsqCV

AERqCV = sum(errorsqCV)/length(groupCol)
AERqCV
# Remark: correct only if we estimate the priors through the 
# sample frequencies!

# AS GENERAL COMMENT, IF THE PERFORMANCES OF TWO MODELS ARE
# COMPARABLE, CHOOSE THE SIMPLEST ONE!

# Plot the partition induced by QDA
x11()
{
	plot(dataset, main='Iris Sepal', xlab='Sepal.Length', 
		 ylab='Sepal.Width', pch=20)
	for(i in 1:g)
	{
		points(dataset[Levels[[i]],], col=rainbow(i), pch=20)	
	}
	legend(min(dataset[,1]), max(dataset[,2]), 
			legend=levels(groupCol), 
			fill=c('red','green','blue'))

	points(QDA$means, col=c('red','green','blue'), pch=4, lwd=2, 
		   cex=1.5)

	x  = seq(min(dataset[,1]), max(dataset[,1]), length=200)
	y  = seq(min(dataset[,2]), max(dataset[,2]), length=200)
	xy = expand.grid(Sepal.Length=x, Sepal.Width=y)

	z  = predict(QDA, xy)$post  
	z1 = z[,1] - pmax(z[,2], z[,3])    
	z2 = z[,2] - pmax(z[,1], z[,3])    
	z3 = z[,3] - pmax(z[,1], z[,2])

	contour(x, y, matrix(z1, 200), levels=0, drawlabels=F, add=T)
	contour(x, y, matrix(z2, 200), levels=0, drawlabels=F, add=T)
	contour(x, y, matrix(z3, 200), levels=0, drawlabels=F, add=T)
}


# NOTE: we have to embedd in the model the missclassification costs
# 		"by hands", since it is not possible to specify them to the 
# 		function qda()
#		However, qda(), as well as lda(), allows to specify the
#		prior probabilities --> we can modify (AND NORMALIZE AGAIN)
#		the priors iot take care of the costs
#
#		R1 = {c(2|1)p1 f1(x) < c(1|2)p2 f2(x)}
#
#		---->
#
#		R1 = {\tilde{p1} f1(x) < \tilde{p2} f2(x)}
#
#		where
#
#		\tilde{p1} = \frac{c(2|1)p1}{c(2|1)p1 + c(1|2)p2}
#
#		\tilde{p2} = \frac{c(1|2)p2}{c(2|1)p1 + c(1|2)p2} = 
#				   = 1 - \tilde{p1}

true = read.table('moneytrue.txt',header=TRUE)
false = read.table('moneyfalse.txt',header=TRUE)

# we impose an order to the labels, which is needed iot correctly 
# assign the priors
# If no order is specified, the labels are assigned in alphabetic
# order
vf = factor(rep(c('true','false'),each=100), 
			levels=c('true','false'))

c_vf = 10		# cost if we accept a false banknote
c_fv = 0.05		# cost if we reject a valid banknote

prior = c(1-0.001,0.001)	# prior probabilities
pv = prior[1]
pf = prior[2]

# Prior modified to account for the misclassification costs
prior_c = c(pv * c_fv / (c_vf * pf + c_fv * pv),
			pf * c_vf / (c_vf * pf + c_fv * pv))

# NOTE: the prior of false after the modification is higher than 
# 		before --> since to missclassify a fals banknote as valid
#		is more expensive than the other way round, we increase
#		the relative prior probability, so that the classifier is
#		more likely to correctly classify it

# QDA
banconote = rbind(true,false)
QDA = qda(banconote, vf, prior=prior_c)

# APER - it is computed by predicting the label on the training set
Qda = predict(QDA)
table(class_true=vf, class_allocated=Qda$class)

APER = 2/100*prior[1]+80/100*prior[2]	# REMARK: THE TRUE PRIOR
										# PROBABILITIES, NOT THE
										# MODIFIED ONES

# Expected economic loss:
80/100*pf*c_vf+2/100*pv*c_fv


# Fisher discriminant analysis in LAB_9



#####################################################################
#							  SVM								   #
###################################################################

library(e1071)

# The classes are not separable
x11()
{
	plot(x, col =ifelse(y==1, 'light blue', 'salmon'), 
    	 pch=19, xlab='x1', ylab='x2', asp=1)
}

# Fit the Support Vector Classifier (kernel = "linear")
# given a cost C
dat = data.frame(x=x, y=as.factor(y))	# the function as.factor() is
										# needed to specify that y
										# is a vector of labels
										
# y~. means that the label y depends on . (all variables)
svmfit = svm(y~., data = dat , kernel = 'linear', cost = 10,
             scale = FALSE )
summary(svmfit)

# non linear kernels
# 1)
#svmfit =svm(y~., data=dat [train ,], kernel ='radial', gamma =1,
#              cost =1)


# 7 support vectors --> (4 3): 4 from the first group and 3 from 
#							   the second

x11()
{
	par(mfrow=c(1,2))
	plot(svmfit , dat, col =c('salmon', 'light blue'), pch=19, asp=1)
}
# NOTE: with crosses, the plot indicates the support vectors; with o
# all the other points

# support vectors are:
svmfit$index

# If we try to change the cost parameter we get more support points
# (higher bias, lower variance)
svmfit = svm(y~., data=dat , kernel ='linear', cost =0.1,
            scale =FALSE )

x11()
{
	plot(svmfit , dat, col =c('salmon', 'light blue'), pch=19, 
		 asp=1)
}

# To set the parameter C we can use the function tune(),
# which is based on cross-validation (10-fold)
set.seed (1)
tune_out = tune(svm ,y~.,data=dat ,kernel = 'linear',
              ranges =list(cost=c(0.001 , 0.01, 0.1, 1,5,10,100) ))
summary(tune_out)

# Extract the best model from the result of tune
bestmod = tune_out$best.model
summary(bestmod)

x11()
{
	plot(bestmod , dat, col =c('salmon', 'light blue'), pch=19, 
		 asp=1)
}

# Prediction for a new observation (command predict())
xtest = matrix(rnorm (20*2) , ncol =2)
ytest = sample(c(-1,1) , 20, rep=TRUE)
xtest[ytest ==1 ,] = xtest[ytest==1,] + 1
testdat = data.frame (x=xtest, y=as.factor(ytest))

ypred = predict(bestmod,testdat)
table(true.label=testdat$y, assigned.label =ypred )



#####################################################################
#							  CLUSTERING						   #
###################################################################

n = dim(dataset)[[1]]

# compute the dissimilarity matrix of the data
# we choose the Euclidean metrics (and then we look at other metrics)
dissimilarity_e = dist(dataset, method='euclidean')
		
# with other metrics:
dissimilarity_m = dist(dataset, method='manhattan')
dissimilarity_c = dist(dataset, method='canberra')

x11()
{
	par(mfrow=c(1,3))
	image(1:n, 1:n, as.matrix(dissimilarity_e), 
		  main='metrics: Euclidean', 
		  asp=1, xlab='i', ylab='j')
	image(1:n, 1:n, as.matrix(dissimilarity_m), 
		  main='metrics: Manhattan', 
		  asp=1, xlab='i', ylab='j')
	image(1:n, 1:n, as.matrix(dissimilarity_c), 
		  main='metrics: Canberra', 
		  asp=1, xlab='i', ylab='j')
}

# --> all the three distances have the same structure --> there are
# two groups that are similar one to the other and one which is 
# really separated
#
# however, a so clear separation is due to the fact that actually data
# are ordered according to the label (since it is a dataset that is
# generated along with a label, which of course is not the case in
# practice, when you have a clustering problem)

# in fact, the data are never ordered according to (unknown) labels
misc = sample(n)
dataset = dataset[misc,]

# hierarchical clustering (Euclidean distance)
ES = hclust(dissimilarity_e, method='single')
EA = hclust(dissimilarity_e, method='average')
EC = hclust(dissimilarity_e, method='complete')

# if we wanted more detailed information on euclidean-complete
# clustering:
names(EC)
EC$merge	# order of aggregation of statistical units / clusters
EC$height	# distance at which we have aggregations
EC$order	# ordering that allows to avoid intersections in the 
			# dendrogram

# plot of the dendrograms
x11()
{
	par(mfrow=c(1,3))
	plot(ES, main='euclidean-single', hang=-0.1, xlab='', 
		 labels=F, cex=0.6, sub='')
	plot(EC, main='euclidean-complete', hang=-0.1, xlab='', 
		 labels=F, cex=0.6, sub='')
	plot(EA, main='euclidean-average', hang=-0.1, xlab='', 
		 labels=F, cex=0.6, sub='')
}

# How to cut a dendrogram?
# We generate vectors of labels through the command cuttree()
# Fix k=2 clusters:
clusterEC = cutree(EC, k=2) # euclidean-complete

clusterES = cutree(ES, k=2) # euclidean-single
clusterEA = cutree(EA, k=2) # euclidean-average

# Let's give a mark to the algorithms: did they aggregate coherently 
# with the dissimilarity matrix or not?

# compute the cophenetic matrices 
cophES = cophenetic(ES)
cophEC = cophenetic(EC)
cophEA = cophenetic(EA)

# compare with dissimilarity matrix (Euclidean distance)
x11()
{
	layout(rbind(c(0,1,0),c(2,3,4)))
	image(as.matrix(dissimilarity_e), main='Euclidean', asp=1 )
	image(as.matrix(cophES), main='Single', asp=1 )
	image(as.matrix(cophEC), main='Complete', asp=1 )
	image(as.matrix(cophEA), main='Average', asp=1 )
}

es = cor(dissimilarity_e, cophES)
ec = cor(dissimilarity_e, cophEC)
ea = cor(dissimilarity_e, cophEA)

c("Eucl-Single"=es,"Eucl-Compl."=ec,"Eucl-Ave."=ea)

# interpret the clusters
table(true_label=groupCol[misc], cluster_label = clusterES)
table(true_label=groupCol[misc], cluster_label = clusterEC)
table(true_label=groupCol[misc], cluster_label = clusterEA)

x11()
{
	plot(dataset, col=ifelse(clusterES==1,'red','blue'), pch=19)
}
x11()
{
	plot(dataset, col=ifelse(clusterEC==1,'red','blue'), pch=19)
}
x11()
{
	plot(iris4, col=ifelse(clusterEA==1,'red','blue'), pch=19)
}


# k-means LAB_11



#####################################################################
#							LINEAR MODELS						   #
###################################################################

### Multiple linear regression

n          = dim(dataset)[[1]]
distance   = dataset$yy
xx1     = dataset$xx
xx2     = dataset$xx^2

# Model:
# yy = beta_0 + beta_1 * xx + beta_2 * xx^2 + Eps
# (linear in the parameters!)

# Assumptions
# 1) Parameter estimation: E(Eps) = 0  and  Var(Eps) = sigma^2 
# 2) Inference:            Eps ~ N(0, sigma^2)


# 1) Estimate of the parameters
# Assumptions: E(Eps) = 0  and  Var(Eps) = sigma^2 

fm = lm(yy ~ xx1 + xx2)

summary(fm) 

fitted(fm)        # y hat
residuals(fm)     # eps hat

coefficients(fm)  # beta_i
vcov(fm)          # cov(beta_i)

fm$rank # order of the model [r+1]
fm$df   # degrees of freedom of the residuals [n-(r+1)]

hatvalues(fm) # h_ii
rstandard(fm) # standardized residuals

s2_err = sum(residuals(fm)^2)/fm$df  # estimate of sigma^2

x11()
{
	plot(dataset, xlab = 'xx', ylab = 'yy', las = 1, 
		 xlim=c(0,30), ylim=c(-5,130))
	x = seq(0,30,by=0.1)
	b = coef(fm)
	lines(x, b[1]+b[2]*x+b[3]*x^2)
}





#####################################################################
#							  SAVE DATA							   #
###################################################################

# matrix on file
write.table(dataset, file = 'filename_out.txt')

# different objects in RData
W <- matrix(data = c(11,12,13,14,15,16), nrow = 2, ncol = 3, byrow = F)
V <- t(W)
a <- 1
save(W,V,a, file = 'variousobjects.RData')