Dataset <- read.table("missing_data.csv", 
                      header=TRUE, stringsAsFactors=TRUE, sep=",", na.strings="NA", dec=".", 
                      strip.white=TRUE)
df = subset(Dataset, select = -c(X,X21022.0.0) )#drop age column to perform mice only on the features
library(mice)
#mice:::find.collinear(Dataset) #Search for highly collinear variables
imputed_Data <- mice(df, m=5, maxit = 5, method = 'rf', seed=666, remove.collinear=FALSE)
complete_data <- complete(imputed_Data,3)
write.csv(complete_data,"imputed.csv",row.names=FALSE)


