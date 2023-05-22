# Biological Bone Age Prediction

## About
The chronological age (CA), which depends on the date of birth, and biological age (BA), which depends on the organ/tissue condition, differs between individuals due to the variation in biological properties and lifestyle behaviors, resulting in differences in the speed of aging. Due to the strong association between aging and deterioration of bone, prediction of biological bone age from bone properties may prove as a viable biomarker for aging.

## Contents
- [Dataset](https://github.com/ChengDHow/Biological-Bone-Age-Prediction/tree/main#dataset)
- [Data Imputation](https://github.com/ChengDHow/Biological-Bone-Age-Prediction/tree/main#data-imputation)
- [Deep learning prediction model](https://github.com/ChengDHow/Biological-Bone-Age-Prediction/tree/main#deep-learning-prediction-model)
- [Evaluation of prediction model](https://github.com/ChengDHow/Biological-Bone-Age-Prediction/tree/main#evaluation-of-prediction-model)

## Dataset

![UKB logo](/assets/images/UKB_logo.jpg)

The dataset used in this project is the [UK Biobank](https://www.ukbiobank.ac.uk/) database. As the database is not publicly available and requires subscription, please contact relevant personnels from the organisation to obtain access rights to the data.

## Data Imputation
In order to reduce the bias and error caused by the large amount of missing data from the dataset, Multivariate Imputation by Chained Equation (MICE) was used to impute the missing data and create a complete dataset for prediction model training. The [MICE package](https://cran.r-project.org/web/packages/mice/index.html) in R was used in this project, which can be installed with the following command:
    
    install.packages('mice')
Run the **`R_mice.R`** script from the **Data_imputation** folder to run MICE imputation and save the results in the destination folder.


## Deep learning prediction model
In this project, the baseline used for model comparison is a linear regression model (**`Baseline_Linear_Regression`**). Various machine learning and deep learning models were tested out and the code for it can be found in the **Prediction_models** folder. The deep learning models are built using [Pytorch](https://pytorch.org/) while the machine learning models are built using [scikit-learn](https://scikit-learn.org/stable/) packages, which can be installed using the following commands:
        
    pip install torch torchvision torchaudio
    pip install -U scikit-learn

When building the prediction model, 80% of the data were used for testing while 20% were used for testing.

## Evaluation of prediction model
A subset of the dataset was extracted with samples that has an observed record of fracture due to simple fall *(0 for no fracture and 1 for fracture)*, which is seen as an indicator of bone deterioration.
<p>The trained model was used to predict the biological bone age of the samples in this subsetted data and the predicted biological bone age was used as a predictor for fracture due to simple fall (outcome) using logistic regression, while controlling for other confounders (gender, ethnicity, smoking status, alcohol consumption and physical activity level).</p>

![logistic regression](/assets/images/Logistic_regression_results.png)

As seen from the logistic regression results, there is a positive and significant association between predicted biological bone age and incidence of fracture from simple fall.

## Citation
The Recurrent Neural Network (RNN) code was completed with reference to the tutorial by [Patrick Loeber](https://github.com/patrickloeber). 

## Author
CHENG Teh How (https://github.com/ChengDHow)

