##############################
# USEFUL FUNCTIONS / SCRIPTS #
##############################

##############################
# COMMON UTILITIES           #
##############################

D.mean   <- sapply(D,mean)
D.cov    <-  cov(D)
D.invcov <- solve(D.cov)

##############################
# PCA                        #
##############################

# PCA (on the covariance matrix)
pc.age <- princomp(age, scores=T, scale = F)
pc.age
summary(pc.age)

# Explained variance
x11()
layout(matrix(c(2,3,1,3),2,byrow=T))
barplot(pc.age$sdev^2, las=2, main='Principal Components', ylab='Variances')
barplot(sapply(age,sd)^2, las=2, main='Original variables', ylab='Variances')
plot(cumsum(pc.age$sdev^2)/sum(pc.age$sde^2), type='b', axes=F, xlab='number of components', ylab='contribution to the total variance', ylim=c(0,1))
abline(h=1, col='blue')
abline(h=0.8, lty=2, col='blue')
box()
axis(2,at=0:10/10,labels=0:10/10)
axis(1,at=1:ncol(age),labels=1:ncol(age),las=2)
#first two components are useful

# Scores
scores.age <- pc.age$scores
scores.age

x11()
layout(matrix(c(1,2),2))
boxplot(age, las=2, col='gold', main='Original variables')
scores.age <- data.frame(scores.age)
boxplot(scores.age, las=2, col='gold', main='Principal components')
#first boxplot has some outliers while the second has not => it's
#clear that we need two PC.

load.age    <- pc.age$loadings
load.age

x11()
par(mar = c(1,4,0,2), mfrow = c(3,1))
for(i in 1:3)barplot(load.age[,i], ylim = c(-1, 1))
#PC1 <=> large counties in term of populations 
#PC2 <=> youngsters > olds

x11()
plot(scores.age[,1],scores.age[,2],type="n",xlab="pc1",ylab="pc2", asp=1)
text(scores.age[,1],scores.age[,2],dimnames(age)[[1]], cex=0.7)

x11()
biplot(pc.age)

##############################
# CONDITIONED PROBABILITY    #
##############################

mu=c(1,1,1)
Sigma=cbind(c(5,3,1),c(3,5,1),c(1,1,1))

# Functions to compute the mean and the covariance matrix of the conditional
# distribution
mu.cond <- function(mu1,mu2,Sig11,Sig12,Sig22,x2)
{
  return(mu1+Sig12%*%solve(Sig22)%*%(x2-mu2))
}

Sig.cond <- function(Sig11,Sig12,Sig22)
{
  Sig21=t(Sig12)
  return(Sig11-Sig12%*%solve(Sig22)%*%Sig21)
}

M.c <- mu.cond(mu1=mu[1:2],mu2=mu[3],Sig11=Sigma[1:2,1:2],Sig12=Sigma[1:2,3],Sig22=Sigma[3,3],x2=1)
Sigma.c <- Sig.cond(Sig11=Sigma[1:2,1:2],Sig12=Sigma[1:2,3],Sig22=Sigma[3,3])

##############################
# NORMALITY TEST             #  
##############################
### Approach 1: look at some linear combinations of the original variables
# we perform univariate tests of normality on the two components
shapiro.test(X[,1])
shapiro.test(X[,2])
# Recall: Shapiro-Wilk test
# H0: X~N 
# test statistics: W=(angular coeff. of the qqline)^2/sample variance
# One can prove that:
# - w<1
# - the empirical distribution under H0 is concentrated on values near 1 
#   (if n=50 more than 90% of the obs. is between 0.95 and 1)
#   (if n=5 more than 90% of the obs. is between 0.81 and 1).
# 
# If the data do NOT come from a Gaussian distribution, the distribution of 
# the test statistics moves toward smaller values:
# Small values of the statistics W give evidence against H0
# => reject H0 for small values of W
# in particular if w < F(w,1-a)
# p value = 0.81 we are very likely to observe sthg like that => it's good.

### Approach 2:
### we consider the Mahalanobis distances of the data from the (sample) mean
### and test if they are a sample from a chi-square distribution

# Recall:
# Theorem: if X~N(mu,Sigma) r.v. in R^p, det(Sigma)>0
#          then d2(X,mu)=(X-mu)'Sigma^-1(X-mu) ~ Chi-sq(p)


x11()
plot(X, asp=1,xlab='X.1',ylab='X.2')

for(prob in (1:14)/15)
  dataEllipse(X, levels = prob , add=T)

x11(width=13)
par(mfrow=c(1,2))
d2 <- mahalanobis(X, colMeans(X), cov(X))   
hist(d2, prob=T, main='Histogram of the Mahalanobis dist.',xlab='d2',ylab='density', col='grey84')
lines(0:2000/100, dchisq(0:2000/100,2), col='blue', lty=2, lwd=2)
qqplot(qchisq((1:n - 0.5)/n, df = 2), d2, main='QQplot of (sample) d2',xlab='theoretical quantiles Chi-sq(2)',
       ylab='sample quantiles')
abline(0, 1)

### we can perform a chi.sq goodness-of-fit test

d2.class <- cut(d2, qchisq((0:10)/10, df = 2))
d2.freq  <- table(d2.class)

chisq.test(x = d2.freq, p = rep(1/10, 10), simulate.p.value = T)
# Test: does the population probabilities (given in x) equal those in p?

# Remark: since the mean and covariance matrix are unknown, we can only
#     have approximate solutions:
#     the Mahalanobis distance is computed with estimates of the mean vector
#     and of the covariance matrix; the sample of distances is not iid

#_______________________________________________________________________________
### Approach 3: test of all the directions simultaneously, by looking at the min
### of the shapiro-wilk statistics

# Basic Idea: I know how to make a test on a single direction => I need to generalize
# to every direction using the fact that X ~ N <=> a'x ~ N....
# We reject H0: X ~ N if we observe a "low" value of W along at least 
# one direction, i.e., if the minimum of W along the direction is "low" 
# Wa statistic of S test for direction a.
# We reject the W_{tilde} = min_a(Wa) iff there exists a direction for which we would reject.
# In this way we have a statistic for our multivariate test

# Looking at all directions, is equivalent to looking at min(W) along
# the directions (test statistic)

# Note: to set the rejection region, we should look at the distribution of 
# of min(W) under (H0) [not on the distribution of W.a for all the directions a]
# --> To see how much we reject globally if we set a threashold
#     alpha=10% at the univariate tests based on W.a see Experiment.R

# Example with our simulated data: we compute the W statistics for all
# the directions. In all the other cases we couldn't do this computation, 
# but we have an explicit expression for W.min which can be used
mcshapiro.test(X)
# Wmin 
# Pvalue = 0.8284 => we can't reject H0
# std dev => we know how many digits we can trust for the result.
# sim <=> #simulations done

##############################
# BOX-COX TRANSFROMATION     #  
##############################


# lambda = 1 => just shifting the data => we're not changing gaussianity.
# For lambda>1: observations <1 are "shrinked", observations >1 are "spread"
# For lambda<1: observations <1 are "spread", observations >1 are "shrinked"
# Compressed/dispersed <=> comparing length of the domain and the one of the codomain
# in the plot of the transformation

### Univariate Box-Cox transformation
lambda.x <- powerTransform(x) 
lambda.x
bc.x <- bcPower(x, lambda.x$lambda)
### Bivariate Box-Cox transformation
lambda <- powerTransform(cbind(x,y))
lambda

BC.x <- bcPower(x, lambda$lambda[1])
BC.y <- bcPower(y, lambda$lambda[2])

##############################
# REMOVING OUTLIERS          #  
##############################

d2 <- matrix(mahalanobis(stiff, colMeans(stiff), cov(stiff)))
# d2 = distance of the point from the center
# here we've used 7.5 as threshold for the Mahalanobis distance
stiff.noout <- stiff[which(d2<7.5),]

##############################
# TEST MULTIVARIATE GAUSSIAN #  
##############################

# Premiss: general rule to perform a test
# 1)  Formulate the test (and test the Gaussian assumption, if needed)
# 2)  Compute the test statistics 
# 3a) Having set the level of the test, verify whether the test statistics 
#     belongs to the region of rejection (i.e., if there is statistical  
#     evidence to reject H0)
# 3b) Compute the p-value of the test

# under h0 we have that T^2 = n*mahalanobis distance(x, mu_0) ~ (n-1)*p/(n-p) F(p,n-p)

#_______________________________________________________________________________
### Test on the mean of level alpha=1%
### H0: mu=mu0 vs H1: mu!=mu0
### with mu0=c(1,0)
###-----------------------------------
mcshapiro.test(x)
# High p-value => we accept it => we can consider the data as gaussian
alpha <- 0.01
mu0 <- c(1,0)

# T2 Statistics
x.T2       <- n * (x.mean-mu0) %*% x.invcov %*% (x.mean-mu0) 
# Radius of the ellipsoid
cfr.fisher <- ((n-1)*p/(n-p))*qf(1-alpha,p,n-p)
# this is the threshold
# Test: 
x.T2 < cfr.fisher   # Rejection region: {x.T2>cfr.fisher}
# (we reject for large values of the T2 statistics)
# we can't reject H0 => accept H0

# Compute the p-value 
P <- 1-pf(x.T2*(n-p)/((n-1)*p), p, n-p)

x11()
xx <- seq(0,40,by=0.05)
plot(xx,df(xx*(n-p)/((n-1)*p),p,n-p),type="l",lwd=2,main='Density F(p,n-p)',xlab='x*(n-p)/((n-1)*p)',ylab='Density')
abline(h=0,v=x.T2*(n-p)/((n-1)*p),col=c('grey','red'),lwd=2,lty=c(2,1))
# The P-value is high because the test statistics is central with respect to 
# its distribution under H0
# => we cannot reject for any reasonable level (we would reject for a level
#    alpha>86%, i.e., if we were allowed to condamn 86% of the innocents!)

dev.off()

# Region of rejection (centred in mu0!)
x11()
plot(x, asp = 1)
ellipse(mu0, shape=x.cov/n, sqrt(cfr.fisher), col = 'blue', lty = 2, center.pch = 4, center.cex=1.5, lwd=2)
points(mu0[1], mu0[2], pch = 4, cex = 1.5, lwd = 2, col ='blue')

# Question:
# where do we expect to find the sample mean with respect to the rejection region?

# We add a red point in correspondence of the sample mean
points(x.mean[1], x.mean[2], pch = 4, cex = 1.5, lwd = 2, col ='red')

# Confidence region
# { m \in R^2 s.t. n * (x.mean-m)' %*% (x.cov)^-1 %*% (x.mean-m) < cfr.fisher }

ellipse(x.mean, x.cov/n, sqrt(cfr.fisher), col = 'red', lty = 2, lwd=2, center.cex=1)

# Remark: the radius and the shape of the ellipse are the same, but the centre changes:
# - Rejection region: the centre is the mean mu0 under H0 (blue ellipse)
# - Confidence region: the centre is the sample mean (red ellipse)
# - Rejection region: {x in R^2: n*(x-mu)'*S^-1*(x-mu) <= (n-1)*p/(n-p) F1-alpha(p,n-p)}
# - Confidence region: {x in R^2: n*(x-x_bar)'*S^-1*(x-x_bar) <= (n-1)*p/(n-p) F1-alpha(p,n-p)}

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

##############################
# TEST ASYMPTOTIC            #  
##############################

### H0: mu=mu0 vs H1: mu!=mu0
### with mu0=c(1,0)
###---------------------------------------------

# Note: we don't need to verify the Gaussian assumption!
# Warning: we are going to use an asymptotic test, but we only have n = 30 data!
# this is thanks to the CLT
# Using asymptotics we don't have anymore the fisher distribution but we have 
# a chi square

mu0   <- c(1,0)

x.T2A   <- n * (x.mean-mu0) %*%  x.invcov  %*% (x.mean-mu0)
cfr.chisq <- qchisq(1-alpha,p)
x.T2A < cfr.chisq
# We accept again H0

# Compute the p-value
PA <- 1-pchisq(x.T2A, p)
# Since we're gaussian we know the true p-value and we see that they are close which
# implies that the approximation is very good.

x11(width=14, height=7)
par(mfrow=c(1,2))
plot(x, asp = 1,main='Comparison rejection regions')
ellipse(mu0, shape=x.cov/n, sqrt(cfr.fisher), col = 'blue', lty = 1, center.pch = 4, center.cex=1.5, lwd=2)
points(mu0[1], mu0[2], pch = 4, cex = 1.5, lwd = 2, col ='blue')
ellipse(mu0, x.cov/n, sqrt(cfr.chisq), col = 'lightblue', lty = 1, center.pch = 4, center.cex=1.5, lwd=2)
points(mu0[1], mu0[2], pch = 4, cex = 1.5, lwd = 2, col ='lightblue')
legend('topleft', c('Exact', 'Asymptotic'),col=c('blue','lightblue'),lty=c(1),lwd=2)

plot(x, asp = 1,main='Comparison of confidence regions')
points(x.mean[1], x.mean[2], pch = 4, cex = 1.5, lwd = 2, col ='red')
ellipse(x.mean, x.cov/n, sqrt(cfr.fisher), col = 'red', lty =1, lwd=2, center.cex=1)
points(x.mean[1], x.mean[2], pch = 4, cex = 1.5, lwd = 2, col ='orange')
ellipse(x.mean, x.cov/n, sqrt(cfr.chisq), col = 'orange', lty = 1, center.pch = 4, center.cex=1.5, lwd=2)
legend('topleft', c('Exact', 'Asymptotic'),col=c('red','orange'),lty=c(1),lwd=2)

# We see that with the same alpha the Asymptotic result is a bit more strict: we
# would reject more data using it even if they are very similar.
# For the same reason we find that the two confidence region are similar but the
# Asymptotic one is smaller
# Be careful this is not a probability region!
# Probability region for X ~ N(mu, Sigma): {x : (x-x_bar)'S^-1(x-x_bar)}
# IMPORTANT: THERE IS NO N BRUNOOOOOOOOOOOOO
# comceptually there is a big difference we can't just say that I was almost there
# the first is a region for the mean, the Probability region is a region for OBSERVATIONS

##############################
# BONFERRONI CI              #  
##############################

k <- p # number of intervals I want to compute (set in advance)!!!!
# IMPORTANT they are 1d studies => we use the student-T statistic as in PAGANONI
# but beware of the fact that the quantile will be 1-alpha/(2*k)
cfr.t <- qt(1-alpha/(2*k),n-1)
Bf <- cbind(inf = x.mean - cfr.t*sqrt(diag(x.cov)/n),
            center = x.mean, 
            sup = x.mean + cfr.t*sqrt(diag(x.cov)/n))
Bf
# Both the intervals contain the mean under H0
# (i.e., mu0 is contained in the rectangular region determined by
# the Bonferroni intervals along the coordinate directions)

# we add the Bonferroni intervals to the plot
rect(Bf[1,1],Bf[2,1],Bf[1,3],Bf[2,3], border='orange', lwd=2)

# IMPORTANT
# Remark: if we wanted to compute additional Bonferroni intervals
# along other directions, we would need to re-compute all the Bonferroni
# intervals with another correction k


##############################
# PAIRED MULTIVARIATE        #  
##############################

effluent <- read.table('effluent.DAT')
colnames(effluent) <- c('BOD_Lab1','SS_Lab1','BOD_Lab2','SS_Lab2')
effluent

x11()
pairs(effluent,pch=19, main='Dataset effluent')

dev.off()

# we compute the sample of differences
D <- data.frame(DBOD=effluent[,1]-effluent[,3], DSS=effluent[,2]-effluent[,4]) 
D

x11()
plot(D, asp=1, pch=19, main='Dataset of Differences')
abline(h=0, v=0, col='grey35')
points(0,0, pch=19, col='grey35')
# DBOD: difference in 'biochemical oxygen demand'
# DSS:  difference in 'suspended solids'
#       measured by the two laboratories

dev.off()

# Now we can proceed as we already know, but working on D

##############################
# TEST FOR REPEATED MEASURES #  
##############################

pressure <- read.table ('pressure.txt', col.names=c('h.0','h.8','h.16','h.24'))
head(pressure)

mcshapiro.test(pressure)

x11()
matplot(t(pressure), type='l')

### question (a)
dim(pressure)

n <- 50
q <- 4

S <- cov(pressure)
M <- sapply(pressure,mean)

#Mathematical framework:
# X_i ~ N_q(mu_i, Sigma)
# H0 <=> Cmu = 0
# H1 <=> Cmu != 0
# T^2 =  n(C(x_bar - 0 ))' (C'SC)^-1 (C(x_bar - 0 ))
# T^2 ~ (n-1)*(q-1)/(n-q+1) F(q-1,n-q+1)
# IMPORTANT
# Please notice that multiplying x_bar by a matrix we change the dimension of x => we change
# the dimension => parameters of the fisher
# sometimes there are contrast matrix whose interpretation is easier


# we build one of the possible contrast matrices to answer
# the question
C <- matrix(c(-1, 1, 0, 0,
              -1, 0, 1, 0,
              -1, 0, 0, 1), 3, 4, byrow=T)
# C has dimension q-1, q
C
# here we are looking at the effects on the pressure
# after 8, 16 and 24 hours from the instant the drug was
# given

# Test: H0: C%*%mu=0 vs H1: C%*%mu!=0
alpha   <- .05
delta.0 <- c(0,0,0)

Md <- C %*% M 
Sd <- C %*% S %*% t(C)
Sdinv <- solve(Sd)

T2 <- n * t( Md - delta.0 ) %*% Sdinv %*% ( Md - delta.0 )

cfr.fisher <- ((q-1)*(n-1)/(n-(q-1)))*qf(1-alpha,(q-1),n-(q-1)) 
# pay attention to the factor!

# T2 is much higher than cfr.fisher => the pvalue will be very small!
P <- 1-pf(T2*(n-(q-1))/((q-1)*(n-1)),(q-1),n-(q-1))

### question (b)

# It is implicitely asking for confidence intervals on the components
# (for the mean of the increments after 8 hours, 16 hours and 24 hours)

# We choose to provide Bonferroni intervals (but we could have chosen
# simultaneous T2 intervals)
k     <- q - 1   # number of increments (i.e., dim(C)[1])
cfr.t <- qt(1-alpha/(2*k),n-1)

# In this way we do the 3 IC in just one line of code
IC.BF <- cbind( Md - cfr.t*sqrt(diag(Sd)/n) , Md, Md + cfr.t*sqrt(diag(Sd)/n) )
# this tell us that after 8h we clearly have higher values and after 16h we clearly
# have lower values while after 24h we can't say that there is a significative difference
# we should also comapre the results with the plots

# Let's compute also the sim. T2 intervals
IC.T2 <- cbind( Md - sqrt(cfr.fisher*diag(Sd)/n) , Md, Md + sqrt(cfr.fisher*diag(Sd)/n) )

x11()
matplot(t(matrix(1:3,3,3)),t(IC.BF), type='b',pch='',xlim=c(0,4),xlab='',ylab='', main='Confidence intervals')
segments(matrix(1:3,3,1),IC.BF[,1],matrix(1:3,3,1),IC.BF[,3], col='orange', lwd=2)
points(1:3, IC.BF[,2], col='orange', pch=16)
points(1:3+.05, delta.0, col='black', pch=16)
segments(matrix(1:3+.1,3,1),IC.T2[,1],matrix(1:3+.1,3,1),IC.T2[,3], col='blue', lwd=2)
points(1:3+.1,IC.T2[,2], col='blue', pch=16)

### what happens if we change the constrast matrix?
Cbis <- matrix(c(-1, 1, 0, 0,
                 0, -1, 1, 0,
                 0, 0, -1, 1), 3, 4, byrow=T)

# in this way we are looking to the mean increment on the effect every 8 hs

Mdbis <- Cbis %*% M 
Sdbis <- Cbis %*% S %*% t(Cbis)
Sdinvbis <- solve(Sdbis)

T2bis <- n * t( Mdbis ) %*% Sdinvbis %*% Mdbis

# compare the test statistics associated with C and Cbis

# What is changed?
# The confidence intervals on the constrasts
# (because we are looking at different contrasts!)

IC.BFbis <- cbind( Mdbis - cfr.t*sqrt(diag(Sdbis)/n) , Mdbis, Mdbis + cfr.t*sqrt(diag(Sdbis)/n) )
IC.T2bis <- cbind( Mdbis - sqrt(cfr.fisher*diag(Sdbis)/n) , Mdbis, Mdbis + sqrt(cfr.fisher*diag(Sdbis)/n) )

#It's a bit more tricky to evaluate these data but it's the same as before

### what if we wanted to verify the hypothesis:
### "the drug decreases the pressure of two units with respect to
### the baseline at both 8 and 16 hs, and its effect vanishes in 24 hs
### from the drug administration"

C <- matrix(c(-1, 1, 0, 0,
              -1, 0, 1, 0,
              -1, 0, 0, 1), 3, 4, byrow=T)
delta.0 <- c(-2,-2,0)

# or
C <- matrix(c(-1, 1, 0, 0,
              0, -1, 1, 0,
              0, 0, -1, 1), 3, 4, byrow=T)
delta.0 <- c(-2,0,2)

Md <- C %*% M 
Sd <- C %*% S %*% t(C)
Sdinv <- solve(Sd)

T2 <- n * t( Md - delta.0 ) %*% Sdinv %*% ( Md - delta.0 )

cfr.fisher <- ((q-1)*(n-1)/(n-(q-1)))*qf(1-alpha,(q-1),n-(q-1))

# pvalue
P <- 1-pf(T2*(n-(q-1))/((q-1)*(n-1)),(q-1),n-(q-1))

##############################
# TEST FOR REPEATED MEASURES #  
##############################
# we compute the sample mean, covariance matrices and the matrix
# Spooled

t1.mean <- sapply(t1,mean)
t2.mean <- sapply(t2,mean)
t1.cov  <-  cov(t1)
t2.cov  <-  cov(t2)
Sp      <- ((n1-1)*t1.cov + (n2-1)*t2.cov)/(n1+n2-2)
# we compare the matrices
list(S1=t1.cov, S2=t2.cov, Spooled=Sp)

# Test H0: mu1=mu2  vs  H1: mu1!=mu2
# i.e.,
# Test H0: mu1-mu2=c(0,0)  vs  H1: mu1-mu2!=c(0,0)

alpha   <- .01
delta.0 <- c(0,0)
Spinv   <- solve(Sp)

T2 <- n1*n2/(n1+n2) * (t1.mean-t2.mean-delta.0) %*% Spinv %*% (t1.mean-t2.mean-delta.0)

cfr.fisher <- (p*(n1+n2-2)/(n1+n2-1-p))*qf(1-alpha,p,n1+n2-1-p)
T2 < cfr.fisher # TRUE: can't reject at 1%

P <- 1 - pf(T2/(p*(n1+n2-2)/(n1+n2-1-p)), p, n1+n2-1-p)
P  
# P-value high (we don't reject at 1%,5%,10%)

# Simultaneous T2 intervals
IC.T2.X1 <- c(t1.mean[1]-t2.mean[1]-sqrt(cfr.fisher*Sp[1,1]*(1/n1+1/n2)), t1.mean[1]-t2.mean[1]+sqrt(cfr.fisher*Sp[1,1]*(1/n1+1/n2)) )
IC.T2.X2 <- c(t1.mean[2]-t2.mean[2]-sqrt(cfr.fisher*Sp[2,2]*(1/n1+1/n2)), t1.mean[2]-t2.mean[2]+sqrt(cfr.fisher*Sp[2,2]*(1/n1+1/n2)) )
IC.T2 <- rbind(IC.T2.X1, IC.T2.X2)
dimnames(IC.T2)[[2]] <- c('inf','sup')                        

##############################
# ANOVA                      #  
##############################

### Model: weigth.ij = mu + tau.i + eps.ij; eps.ij~N(0,sigma^2)
### Test:
### H0: tau.1 = tau.2 = tau.3 = tau.4 = tau.5 = tau.6 = 0
### H1: (H0)^c
### i.e.,
### H0: The feed supplements don't have effect
###     (= "chickens belong to a single population")
### H1: At least one feed supplement has effect
###     (= "chickens belong to 2, 3, 4, 5 or 6 populations")

# We must be very rigorous in the modelling of ANOVA
# mu, tau_i are constants.
# H1 <=> there exists a treatement that has an effect

# This is a case of one-way ANOVA: one variable (weight) observed 
# over g=6 levels (feed)
n       <- length(feed)      # total number of obs.
ng      <- table(feed)       # number of obs. in each group
treat   <- levels(feed)      # levels of the treatment
g       <- length(treat)     # number of levels (i.e., of groups)

# Assumptions: e_ij must be iid gaussian and homoschedastic
# we can try to estimate e_ij and verify these assumptions
# But the most common way to proceed is to check that the orginal
# X are gaussians with mean mu + t_i and same variance (matrix) sigma (Sigma)
# we chose the second way
# If we test on the whole population we will never find sthg gaussian.
# This actually is evident since we're summing up different things that
# we would like to be different.

### verify the assumptions:
# 1) normality (univariate) in each group (6 tests)
Ps <- c(shapiro.test(weight[ feed==treat[1] ])$p,
        shapiro.test(weight[ feed==treat[2] ])$p,
        shapiro.test(weight[ feed==treat[3] ])$p,
        shapiro.test(weight[ feed==treat[4] ])$p,
        shapiro.test(weight[ feed==treat[5] ])$p,
        shapiro.test(weight[ feed==treat[6] ])$p) 

# As usual high p-values for shapiro.test means gaussian :)

# 2) same covariance structure (= same sigma^2)
Var <- c(var(weight[ feed==treat[1] ]),
         var(weight[ feed==treat[2] ]),
         var(weight[ feed==treat[3] ]),
         var(weight[ feed==treat[4] ]),
         var(weight[ feed==treat[5] ]),
         var(weight[ feed==treat[6] ])) 

# test of homogeneity of variances
# We use the bartlett.test 
# H0: sigma.1 = sigma.2 = sigma.3 = sigma.4 = sigma.5 = sigma.6 
# H1: there exist i,j s.t. sigma.i!=sigma.j
bartlett.test(weight, feed)
# we see that the p-value is relatively high => we can't reject h0
# we can assume now that the variances are all the same

### One-way ANOVA 
###-----------------

# Analysis Of Variance
# Syntax: aov(data ~ treatement)
# this is a general syntax in R we use always data ~ response variable
# when we do linear regression/ classification we'll use same syntax

fit <- aov(weight ~ feed)

summary(fit)

### How to read the summary:
#              Df   Sum Sq      Mean Sq      F value     Pr(>F)    
#  treat      (g-1) SStreat  SStreat/(g-1)  Fstatistic  p-value [H0: tau.i=0 for every i]
#  Residuals  (n-g) SSres     SSres/(n-g)                    
#---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1 
###

### We reject the test, i.e., we have evidence to state that the  
### treatment (feed supplement) has an effect on the growth rate
### of chicken
### Which supplement is responsible for this? To see this, we need to 
### do g*(g-1)/2 comparisons. <=> all possible combinations
### We use Bonferroni
k <- g*(g-1)/2

Media   <- mean(weight) #<=> gloabal mean
Mediag  <- tapply(weight, feed, mean) #<=> mean within the group

SSres <- sum(residuals(fit)^2)

S <- SSres/(n-g) #estimator for sigma (Sigma)

# Example: CI for the difference "casein - horsebean"
alpha <-0.05
paste(treat[1],"-",treat[2])
as.numeric(c(Mediag[1]-Mediag[2] - qt(1-alpha/(2*k), n-g) * sqrt( S * ( 1/ng[1] + 1/ng[2] )),
             Mediag[1]-Mediag[2] + qt(1-alpha/(2*k), n-g) * sqrt( S * ( 1/ng[1] + 1/ng[2] ))))
# 0 is not inside the interval => we have a strong evidence 
# CI for all the differences
ICrange=NULL
for(i in 1:(g-1)) {
  for(j in (i+1):g) {
    print(paste(treat[i],"-",treat[j]))        
    print(as.numeric(c(Mediag[i]-Mediag[j] - qt(1-alpha/(2*k), n-g) * sqrt( S * ( 1/ng[i] + 1/ng[j] )),
                       Mediag[i]-Mediag[j] + qt(1-alpha/(2*k), n-g) * sqrt( S * ( 1/ng[i] + 1/ng[j] )))))
    ICrange=rbind(ICrange,as.numeric(c(Mediag[i]-Mediag[j] - qt(1-alpha/(2*k), n-g) * sqrt( S * ( 1/ng[i] + 1/ng[j] )),
                                       Mediag[i]-Mediag[j] + qt(1-alpha/(2*k), n-g) * sqrt( S * ( 1/ng[i] + 1/ng[j] )))))
  }}

# There is a tricky point: In hypothesis test we can't use transitivity
# if with a test we have concluded that A = B and B = C we can't conclude that
# A = C. Because in fact what we're doing are weak equalities <=> weak statements
# we don't "accept" h1 we just don't reject it which is a way weaker result. 

# Let's change the criterion to control the univariate rejection 
# (multiple testing)
k <- g*(g-1)/2
S <- SSres/(n-g)

# We build k confidence intervals, each of level alpha (I'm not including the Bonferroni correction)
Auni <- matrix(0,6,6)
for(i in 1:6) {
  for(j in i:6) {
    Auni[i,j] <- Mediag[i]-Mediag[j] + qt(1-alpha/2, n-g) * sqrt( S * ( 1/ng[i] + 1/ng[j] ) )}
  for(j in 1:i) {
    Auni[i,j] <- Mediag[j]-Mediag[i] - qt(1-alpha/2, n-g) * sqrt( S * ( 1/ng[i] + 1/ng[j] ) )}
  Auni[i,i]     <- 0
}

# We compute the p-values of the univariate tests
# watch out it's UNIVARIATE 
# Matrix of tests for the difference between all the pairs 
P <- matrix(0,6,6)
for(i in 1:6) {
  for(j in i:6) {
    P[i,j] <- (1-pt(abs((Mediag[i]-Mediag[j]) / sqrt( S * ( 1/ng[i] + 1/ng[j] ) ) ), n-g))*2}
  for(j in 1:i) {
    P[i,j] <- (1-pt(abs((Mediag[i]-Mediag[j]) / sqrt( S * ( 1/ng[i] + 1/ng[j] ) ) ), n-g))*2}
  P[i,i]     <- 0
}

# Vector of p-values
p <- c(P[1, 2:6], P[2, 3:6], P[3, 4:6], P[4, 5:6], P[5, 6])

# Bonferroni correction
p.bonf <- p.adjust(p, 'bonf')
#it takes as input the vector of p-values and the type of correction
# before the correction we were rejecting 12/15 now we are rejecting only 8/15
# this correction maybe is too strict
p.bonf-ifelse(p*15>1,1,p*15) # p.bonf is actually p*k

# Correction according to the false discovery rate (Benjamini-Hockberg)
q.fdr <- p.adjust(p, 'fdr')

##############################
# MANOVA                     #  
##############################

### Model: X.ij = mu + tau.i + eps.ij; eps.ij~N_p(0,Sigma), X.ij, mu, tau.i in R^4
### Test:
### H0: tau.1 = tau.2 = tau.3  = (0,0,0)'
### H1: (H0)^c
### that is
### H0: The membership to an iris species hasn't any significant effect on the mean
###     of X.ij (in any direction of R^4) 
### H1: There exists at least one direction in R^4 along which at least two species
###     have some feature significantly different
# Claims that we want to do are on the tau_i as usual. For instance in our 
# case at least the red group play a key role
# We still need to verifiy our assumptions.
# Gaussianity assumption <=> elliptical clouds
# if data come from the same covariances we should have similar ellipses
# about the covariance assumption we will close our eyes and go on anyway. 
# we can perform some test at the end and try to perform some transformation

# This is a case of one-way MANOVA: four variables (seplen, sepwid, petlen, petwid)
# observed over g=3 levels (setosa,versicolor,virginica)
n1 <- length(i1)
n2 <- length(i2)
n3 <- length(i3)
n  <- n1+n2+n3

g  <- length(levels(species.name))
p  <- 4

# 2) same covariance structure (= same covariance matrix Sigma)
S  <-  cov(iris4)
S1 <-  cov(iris4[i1,])
S2 <-  cov(iris4[i2,])
S3 <-  cov(iris4[i3,])

# Note: We can verify the assumptions a posteriori on the residuals of the estimated 
#       model 

### One-way MANOVA 
###----------------

fit <- manova(as.matrix(iris4) ~ species.name)
summary.manova(fit,test="Wilks")   # Exact tests for p<=2 or g<=3 already implemented in R
# Note: since g=3 the test is exact 
#       (cfr. JW pag.300) 

### Reject the test, i.e., we have statistical evidence to state that
### the factor "iris species" has an effect on the mean features 
### of the flowers
### Who's the responsible for this?

# we can conclude that there is AT LEAST ONE DIRECTION along which
# AT LEAST 2 GROUPS we can reject.
# we are in a 4 variate space => lots of directions
# we need one direction and 2 groups to verify this result.
# We can try to use the axis direction => univariate => ANOVA
# Again this is not the only choice !!!
# we could also look for the worst direction as we did in lab 7


### Via ANOVA: for each of the p=4 variables we perform an ANOVA test
###            to verify if the membership to a group has influence
###            on the mean of the variable (we explore separately the
###            4 axes directions in R^4)
summary.aov(fit)

# Each of the 4 variables is significantly influenced by the  
# factor species
# Note. This analysis does NOT say: 
#       a) which group differ
#       b) with respect to which variables the groups in (a) differ
# => As for the ANOVA, we build confidence intervals (many more!)

### Via Bonferroni
alpha <- 0.05
k <- p*g*(g-1)/2
qT <- qt(1-alpha/(2*k), n-g)

W <- summary.manova(fit)$SS$Residuals
m  <- sapply(iris4,mean)         # estimates mu
m1 <- sapply(iris4[i1,],mean)    # estimates mu.1=mu+tau.1
m2 <- sapply(iris4[i2,],mean)    # estimates mu.2=mu+tau.2
m3 <- sapply(iris4[i3,],mean)    # estimates mu.3=mu+tau.3

inf12 <- m1-m2 - qT * sqrt( diag(W)/(n-g) * (1/n1+1/n2) )
sup12 <- m1-m2 + qT * sqrt( diag(W)/(n-g) * (1/n1+1/n2) )
inf13 <- m1-m3 - qT * sqrt( diag(W)/(n-g) * (1/n1+1/n3) )
sup13 <- m1-m3 + qT * sqrt( diag(W)/(n-g) * (1/n1+1/n3) )
inf23 <- m2-m3 - qT * sqrt( diag(W)/(n-g) * (1/n2+1/n3) )
sup23 <- m2-m3 + qT * sqrt( diag(W)/(n-g) * (1/n2+1/n3) )

CI <- list(setosa_versicolor=cbind(inf12, sup12), setosa_virginica=cbind(inf13, sup13), versicolor_virginica=cbind(inf23, sup23))

##############################
# 2 WAY ANOVA                #  
##############################

##### (g = 2, b = 2, p = 1)
#####----------------------

### Problem 4 of 14/09/06
###------------------------
# In a small village in Switzerland there are two gas stations:
# one of Esso and one of Shell. Both sell either gasoline 95 octanes
# and 98 octanes. 
# A young statistician wants to find out which is the best gas station
# and the best gasoline to refuel his car, in order to maximize the 
# number of kilometers covered with a single refueling.
# After 8 refuelings, the measured performances are:
# km/l  : (18.7, 16.8, 20.1, 22.4, 14.0, 15.2, 22.0, 23.3)
# distr.: ('Esso','Esso','Esso','Esso','Shell','Shell','Shell','Shell')
# benz. : ('95','95','98','98','95','95','98','98')
# (a) Via a two-way ANOVA identify which is the best station and the
#     best gasoline for the young statistician to refuel his car
# (b) Is there an interaction between the gas station and the gasoline?

### Variables: distance covered [km/l]
### factor1:   Gas station   (0=Esso, 1=Shell)
### factor2:   Gasoline        (0=95,   1=98)
### Balanced design

km          <- c(18.7, 16.8, 20.1, 22.4, 14.0, 15.2, 22.0, 23.3)
distr       <- factor(c('Esso','Esso','Esso','Esso','Shell','Shell','Shell','Shell'))
benz        <- factor(c('95','95','98','98','95','95','98','98'))
distr_benz  <- factor(c('Esso95','Esso95','Esso98','Esso98','Shell95','Shell95','Shell98','Shell98'))
g <- length(levels(distr))
b <- length(levels(benz))
n <- length(km)/(g*b)

M           <- mean(km)
Mdistr      <- tapply(km,      distr, mean)
Mbenz       <- tapply(km,       benz, mean)
Mdistr_benz <- tapply(km, distr_benz, mean)

### Model with interaction (complete model): 
### X.ijk = mu + tau.i + beta.j + gamma.ij + eps.ijk; eps.ijk~N(0,sigma^2), 
###     i=1,2 (effect station), j=1,2 (effect gasoline)
# unless we're not told differently we start with the whole model
# and we try to remove the interaction because if we're able to do this we
# can put the SS_int and SS_res together and we would maximize the dofs of
# the residuals to better estimate covariance matrix
# this is know as the high-hierarchical principle:
# if we want to keep a higher degree variable we must preserve the lower degree
# one but not the opposite. In this case interactions are of order 2 => we can
# discard them :)
fit.aov2.int <- aov(km ~ distr + benz + distr:benz)
summary.aov(fit.aov2.int)
# Issue: only 1 dof for each factor and only 4 for the residual.
# even if the p-value of the interaction is more or less 5% we will reject it
# to gain 1 dof in the residuals.
# We can remove one part at the time we can't remove a lot of them.

### Test:
### 1) H0: gamma11 = gamma.12 = gamma.21 = gamma.22 = 0    vs   H1: (H0)^c
###    i.e.,
###    H0: There is no significant interaction between the factors station
###        and gasoline in terms of performances
###    H1: There exists a significant interaction between the factors station 
###        and gasoline in terms of performances
###
### 2) H0: tau.1 = tau.2 = 0    vs   H1: (H0)^c
###    i.e.,
###    H0: The effect "gas station" doesn't significantly influence performances 
###    H1: The effect "gas station" significantly influences performances
###
### 3) H0: beta.1 = beta.2 = 0    vs   H1: (H0)^c
###    i.e.,
###    H0: The effect "gasoline" doesn't significantly influence performances
###    H1: The effect "gasoline" significantly influences performances

# Test 1): Let's focus on the row of the summary distr:benz :
#             Df Sum Sq Mean Sq F value  Pr(>F)
# distr:benz   1  10.35   10.35   6.884 0.05857 .
# The P-value of test 1) is 0.05857. Reject at 10%, dont't reject at 1%,5% -> ?

# Test 2): Let's focus on the row of the summary distr:
#             Df Sum Sq Mean Sq F value  Pr(>F)
# distr        1   1.53    1.53   1.018 0.37001   
# The P-value of test 2) is 0.37001. Don't reject at 10%, 5%, 1% -> not significant

# Test 3): Let's focus on the row of the summary benz:
#             Df Sum Sq Mean Sq F value  Pr(>F)
# benz         1  66.70   66.70  44.357 0.00264 **
# The P-value of test 3) is 0.00264. Reject at 10%, 5%, 1% -> significant

# Point b)
# From test 1): We don't have strong evidence that the interaction has effect
# => try to remove the interaction term and estimate the model without interaction

### Additive model: 
### X.ijk = mu + tau.i + beta.j + eps.ijk; eps.ijk~N(0,sigma^2), 
###     i=1,2 (effect station), j=1,2 (effect gasoline)
fit.aov2.ad <- aov(km ~ distr + benz)
summary.aov(fit.aov2.ad)
# Remark: by removing the interaction, the residual degrees of freedom increase! 

# Test: 2bis) H0: tau.1 = tau.2 = 0    vs   H1: (H0)^c
# From the summary:
#             Df Sum Sq Mean Sq F value  Pr(>F)   
# distr        1   1.53    1.53   0.468 0.52440
# The P-value of test 2bis) is 0.52440. Don't reject at 10%, 5%, 1% -> not significant

# Test: 3bis) H0: beta.1 = beta.2 = 0    vs   H1: (H0)^c
# From the summary:
#             Df Sum Sq Mean Sq F value  Pr(>F)   
# benz         1  66.70   66.70  20.378 0.00632 **
# The P-value of test 2bis) is 0.00632. Dont' reject at 10%, 5%, 1% -> significant

### Note: These aren't the only tests we can do!
### Example: global test for the significance of the two treatments 
###          (model without interaction)
SSdistr <- sum(n*b*(Mdistr - M)^2)              # or from the summary: 1.53    
SSbenz  <- sum(n*g*(Mbenz  - M)^2)              # or from the summary: 66.70
SSres   <- sum((km - M)^2) - (SSdistr+SSbenz)   # or from the summary: 16.37

Ftot      <- ( (SSdistr + SSbenz) / ((g-1)+(b-1)))/(SSres / (n*g*b-g-b+1))
Ptot      <- 1 - pf(Ftot, (g-1)+(b-1), n*g*b-g-b+1) # attention to the dgf!

# Test 2bis): there is no evidence that the factor "gas station" have
#             effect on the performances (don't reject at any reasonable 
#             level [high p-value!])
# => we remove the variable "station" and reduce to a one-way ANOVA

### Reduced additive model (ANOVA one-way, b=2): 
### X.jk = mu + beta.j + eps.jk; eps.jk~N(0,sigma^2), 
###     j=1,2 (effect gasoline)
fit.aov1 <- aov(km ~ benz)
summary.aov(fit.aov1)
# this is now a one way ANOVA 

SSres=sum(residuals(fit.aov1)^2)

### Interval at 90% for the differences (reduced additive model)
### [b=2, thus one interval only]
IC <- c(diff(Mbenz) - qt(0.95, (n*g-1)*b) * sqrt(SSres/((n*g-1)*b) *(1/(n*g) + 1/(n*g))), 
        diff(Mbenz) + qt(0.95, (n*g-1)*b) * sqrt(SSres/((n*g-1)*b) *(1/(n*g) + 1/(n*g))))
names(IC) <- c('Inf', 'Sup')

### Note: we should have verified the hypotheses of normality and variance
###       homogeneity for the complete model, but with only 2 data for each
###       group we can't perform the tests.
### => we verify the assumptions on the reduced model (one-way ANOVA)
# 1) normality (univariate) in each groups (2 tests)
Ps <- c(shapiro.test(km[ benz==levels(benz)[1] ])$p,
        shapiro.test(km[ benz==levels(benz)[2] ])$p)

# 2) homogeneity of variances
bartlett.test(km, benz)

##############################
# 2 WAY MANOVA               #  
##############################


##############################
# LDA VS QDA ASSUMPTIONS     #  
##############################

shapiro.test(sardinaa)
shapiro.test(sardinai)
var.test(sardinaa, sardinai)
t.test(sardinaa, sardinai, var.eq=T)

##############################
# LDA                        #  
##############################

lda.iris <- lda(iris2, species.name)
lda.iris

# "coefficients of linear discriminants" and "proportion of trace":
# Fisher discriminant analysis. 
# In particular:
# - coefficients of linear discriminants: versors of the canonical directions
#   [to be read column-wise]
# - proportion of trace: proportion of variance explained by the corresponding 
#   canonical direction
# it's morally like a PCA on the discrimination but these direction are not orthogonal
# since this is not a PCA.
names(lda.iris)

Lda.iris <- predict(lda.iris, iris2)
#Lda.iris
names(Lda.iris)

# Compute the APER
Lda.iris$class   # assigned classes
species.name     # true labels
table(class.true=species.name, class.assigned=Lda.iris$class)

errori <- (Lda.iris$class != species.name)
errori
sum(errori)
length(species.name)

APER   <- sum(errori)/length(species.name)
APER

# Compute the estimate of the AER by cross-validation 
LdaCV.iris <- lda(iris2, species.name, CV=TRUE)  # specify the argument CV
# the default with CV=TRUE is a leave 1 out CV

LdaCV.iris$class
species.name
table(classe.vera=species.name, classe.allocataCV=LdaCV.iris$class)

erroriCV <- (LdaCV.iris$class != species.name)
erroriCV
sum(erroriCV)

AERCV   <- sum(erroriCV)/length(species.name)
AERCV

##############################
# QDA                        #  
##############################

qda.iris <- qda(iris2, species.name)
qda.iris
Qda.iris <- predict(qda.iris, iris2)
#Qda.iris

# compute the APER
Qda.iris$class
species.name
table(classe.vera=species.name, classe.allocata=Qda.iris$class)

erroriq <- (Qda.iris$class != species.name)
erroriq

APERq   <- sum(erroriq)/length(species.name)
APERq
# Remark: correct only if we estimate the priors through the sample frequencies!

# Compute the estimate of the AER by cross-validation 
QdaCV.iris <- qda(iris2, species.name, CV=T)
QdaCV.iris$class
species.name
table(classe.vera=species.name, classe.allocataCV=QdaCV.iris$class)

erroriqCV <- (QdaCV.iris$class != species.name)
erroriqCV

AERqCV   <- sum(erroriqCV)/length(species.name)
AERqCV

# Plot the partition induced by QDA
x11()
plot(iris2, main='Iris Sepal', xlab='Sepal.Length', ylab='Sepal.Width', pch=20)
points(iris2[i1,], col='red', pch=20)
points(iris2[i2,], col='green', pch=20)
points(iris2[i3,], col='blue', pch=20)
legend(min(iris[,1]), max(iris[,2]), legend=levels(species.name), fill=c('red','green','blue'))

points(qda.iris$means, col=c('red','green','blue'), pch=4, lwd=2, cex=1.5)

x  <- seq(min(iris[,1]), max(iris[,1]), length=200)
y  <- seq(min(iris[,2]), max(iris[,2]), length=200)
xy <- expand.grid(Sepal.Length=x, Sepal.Width=y)

z  <- predict(qda.iris, xy)$post  
z1 <- z[,1] - pmax(z[,2], z[,3])    
z2 <- z[,2] - pmax(z[,1], z[,3])    
z3 <- z[,3] - pmax(z[,1], z[,2])

contour(x, y, matrix(z1, 200), levels=0, drawlabels=F, add=T)
contour(x, y, matrix(z2, 200), levels=0, drawlabels=F, add=T)
contour(x, y, matrix(z3, 200), levels=0, drawlabels=F, add=T)

open3d()
points3d(iris2[i1,1], iris2[i1,2], 0, col='red', pch=15)
points3d(iris2[i2,1], iris2[i3,2], 0, col='green', pch=15)
points3d(iris2[i3,1], iris2[i2,2], 0, col='blue', pch=15)
surface3d(x,y,matrix(dmvnorm(xy, m1, S1) / 3, 50), alpha=0.4, color='red')
surface3d(x,y,matrix(dmvnorm(xy, m2, S2) / 3, 50), alpha=0.4, color='green', add=T)
surface3d(x,y,matrix(dmvnorm(xy, m3, S3) / 3, 50), alpha=0.4, color='blue', add=T)
box3d()

##############################
# KNN                        #  
##############################

x11()
plot(iris2, main='Iris.Sepal', xlab='Sepal.Length', ylab='Sepal.Width', pch=15)
points(iris2[i1,], col=2, pch=15)
points(iris2[i3,], col=4, pch=15)
points(iris2[i2,], col=3, pch=15)
legend(min(iris[,1]), max(iris[,2]), legend=levels(species.name), fill=c(2,3,4))

x  <- seq(min(iris[,1]), max(iris[,1]), length=200)
y  <- seq(min(iris[,2]), max(iris[,2]), length=200)
xy <- expand.grid(Sepal.Length=x, Sepal.Width=y)

iris.knn <- knn(train = iris2, test = xy, cl = iris$Species, k = k)

z  <- as.numeric(iris.knn)

contour(x, y, matrix(z, 200), levels=c(1.5, 2.5), drawlabels=F, add=T)

##############################
# MODIFYING PRIORS           #  
##############################

c.vf <- 10
c.fv <- 0.05

prior <- c(1-0.001,0.001)
pv <- prior[1]
pf <- prior[2]

# Prior modified to account for the misclassification costs
prior.c <- c(pv*c.fv/(c.vf*pf+c.fv*pv),pf*c.vf/(c.vf*pf+c.fv*pv))
prior.c

##############################
# SVM                        #  
##############################


# Fit the Support Vector Classifier (kernel = "linear")
# given a cost C
dat=data.frame(x=x, y=as.factor (y))
svmfit =svm(y~., data=dat , kernel ='linear', cost =10,
            scale =FALSE )
summary(svmfit)
# Number of Support Vectors:  7 is the number of points that are actually used
# since it depends only on those points SVM is generallt more stable than LDA.

x11()
par(mfrow=c(1,2))
plot(svmfit , dat, col =c('salmon', 'light blue'), pch=19, asp=1)

# support vectors are indicated with crosses
# they are:
svmfit$index

# If we try to change the cost parameter we get more support points
# (higher bias, lower variance)
svmfit=svm(y~., data=dat , kernel ='linear', cost =0.1,
           scale =FALSE )
plot(svmfit , dat, col =c('salmon', 'light blue'), pch=19, asp=1)

# Setting the cost <=> playing with bias-variance trade off
# high cost => 

# To set the parameter C we can use the function tune(),
# which is based on cross-validation (10-fold)
set.seed (1)
# we should set a seed since CV uses a random partition => we need to be able
# to repeat it
tune.out=tune(svm ,y~.,data=dat ,kernel = 'linear',
              ranges =list(cost=c(0.001 , 0.01, 0.1, 1,5,10,100) ))

# Suggestion : use a cost vector with different order of magnitude and then we can refine it.

# Extract the best model from the result of tune
bestmod =tune.out$best.model
summary(bestmod)

plot(bestmod , dat, col =c('salmon', 'light blue'), pch=19, asp=1)

# Prediction for a new observation (command predict())
xtest=matrix(rnorm (20*2) , ncol =2)
ytest=sample(c(-1,1) , 20, rep=TRUE)
xtest[ytest ==1 ,]= xtest[ytest ==1,] + 1
testdat=data.frame (x=xtest , y=as.factor (ytest))

ypred=predict(bestmod,testdat)
table(true.label=testdat$y, assigned.label =ypred )
