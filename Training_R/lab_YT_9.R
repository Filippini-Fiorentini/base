source("clean.R")

####################
# Lab 9 on YT Part 1
####################

#PCA
dimnames(USArrests)
apply(USArrests, 2, mean)
apply(USArrests, 2, var)


#We decide to standardize variables since they have different variances and then apply PCA
#and prcomp do everything together
pca.out = prcomp(USArrests, scale = TRUE)
pca.out #to see the components of the PC's
names(pca.out)
#We can now do a biplot that will show in red the loading vectors and in black the observations
biplot(pca.out, scale = 0, cex = 6)


####################
# Lab 9 on YT Part 2
####################

#K-means clustering
#Example in 2D so that we can represent it
#We create some fake data
set.seed(101)
x = matrix(rnorm(100,2), 100, 2)
# I want to generate also some random mean with a bigger std deviation
xmean = matrix(rnorm(8, sd = 4), 4, 2)
# randomly assign some indices
which = sample(1:4, 100, replace = TRUE)
x = x +  xmean[which,]
plot(x, col = which, pch = 19)

#nstart = 15 <=> tries with 15 different random starts
km.out = kmeans(x, 5, nstart = 15)
km.out
#plot clusters found
plot(x, col = km.out$cluster, cex = 2, pch = 1, lwd = 2)
#real clusters
points(x, col = which, pch = 19) #order is not the same => we modify it
points(x, col = c(4,3,2,1), pch = 19)
####################
# Lab 9 on YT Part 3
####################

#Hierarchical clustering
#we use the same random data as before 
#hclust is a tool to perform Hierarchical clustering

hc.complete = hclust(dist(x), method = "complete")
plot(hc.complete)
hc.single = hclust(dist(x), method = "single")
plot(hc.single) 
hc.average = hclust(dist(x), method = "average")
plot(hc.average) 

#we decide that the first one is the best one and we use the fct
#cutree that cuts at the level (4) and gives back a vector of assignements
hc.cut = cutree(hc.complete, 4)
table(hc.cut, which)
table(hc.cut, km.out$cluster)
plot(hc.complete, labels = which)

#Rmd document can be used to generate html document with the results => very useful
#but the code must be written inside ```{r} djahsdks ```