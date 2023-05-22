#Script ran in R commander
Dataset <- read.table("fracture_onetp_pred_age.csv", header=TRUE, stringsAsFactors=TRUE,
   sep=",", na.strings="NA", dec=".", strip.white=TRUE)
Dataset_filtered <- subset(Dataset, subset=Fracture >= 0)
GLM.1 <- glm(Fracture ~ Alcohol + Ethnicity + Exercise + Gender + Predicted_age + Smoking, family=binomial(logit), 
  data=Dataset_filtered)
summary(GLM.1)
exp(coef(GLM.1))  # Exponentiated coefficients ("odds ratios")

