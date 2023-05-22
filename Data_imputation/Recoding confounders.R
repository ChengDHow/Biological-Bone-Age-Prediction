
Dataset <- read.table("filtered_onetp.csv", header=TRUE, 
  stringsAsFactors=FALSE, sep=",", na.strings="NA", dec=".", strip.white=TRUE)
Dataset <- within(Dataset, {
  X21000.0.0 <- Recode(X21000.0.0, 
  'c(1,1001,3001,1002,3002,3,1003,3003,3004,5,2003) = 1;c( 2001,4001,2,2002,4002,4003,4,2004,6,-1,-3) = 0;', 
  as.factor=FALSE, to.value="=", interval=":", separator=";")
})
library(abind, pos=17)
library(e1071, pos=18)
numSummary(Dataset[,"X21000.0.0", drop=FALSE], statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,
  .5,.75,1))
Dataset <- within(Dataset, {
  X1558.0.0 <- Recode(X1558.0.0, 'c(5,6) = 0;c( 1,2,3,4,-3) = 1;', as.factor=FALSE, to.value="=", interval=":", 
  separator=";")
})
numSummary(Dataset[,"X1558.0.0", drop=FALSE], statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,.5,
  .75,1))
Dataset <- within(Dataset, {
  X1558.1.0 <- Recode(X1558.1.0, 'c(5,6) = 0;c( 1,2,3,4,-3) = 1; ;', as.factor=FALSE, to.value="=", interval=":", 
  separator=";")
})
Dataset <- within(Dataset, {
  X1558.2.0 <- Recode(X1558.2.0, 'c(5,6) = 0;c( 1,2,3,4,-3) = 1; ; ;', as.factor=FALSE, to.value="=", interval=":",
   separator=";")
})
Dataset <- within(Dataset, {
  X20116.0.0 <- Recode(X20116.0.0, '0=0;c( 1,2,-3) = 1; ; ; ;', as.factor=FALSE, to.value="=", interval=":", 
  separator=";")
})
numSummary(Dataset[,"X20116.0.0", drop=FALSE], statistics=c("mean", "sd", "IQR", "quantiles"), quantiles=c(0,.25,
  .5,.75,1))
Dataset <- within(Dataset, {
  X20116.1.0 <- Recode(X20116.1.0, '0=0;c( 1,2,-3) = 1; ; ; ; ;', as.factor=FALSE, to.value="=", interval=":", 
  separator=";")
})
Dataset <- within(Dataset, {
  X20116.2.0 <- Recode(X20116.2.0, '0=0;c( 1,2,-3) = 1; ; ; ; ; ;', as.factor=FALSE, to.value="=", interval=":", 
  separator=";")
})

write.csv(Dataset,"onetp_confounders_recoded.csv",row.names=FALSE)


