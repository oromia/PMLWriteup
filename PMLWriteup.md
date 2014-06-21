# Machine Learning
### Weight Lifting Prediction Using Accelerometer Data


## Executive Summary
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.

## Data Processing

Data for this project come from thttp://groupware.les.inf.puc-rio.br/har. We Load both datasets.

```r
raw_training <- read.csv("pmltraining.csv")
raw_testing <- read.csv("pmltesting.csv")
```



## Steps 
1. Set Aside validation Sets.
2. Remove indicators with near zero variance.
3. Eliminate highly correlated predictors.
4. Impute missing values.
5. Build Random Forest Model
6. Perform Cross Validation
7. Perform Confusion Matrix
8. Test Prediction Results

#### Set Aside Validation Sets
We do this by partitioning our data to two, training and validation set.


```
Loading required package: lattice
Loading required package: ggplot2
```


#### Remove indicators with near zero variance.
These are uninformative predictors in our dataset

```r
library(caret)
nzv <- nearZeroVar(training)

training <- training[-nzv]
testing <- testing[-nzv]
raw_testing <- raw_testing[-nzv]
```


#### Impute Missing Values
We convert our column to numeric type for better prediction and impute missing values. This will help us avid miscalculation.


```r
num_features_idx = which(lapply(training, class) %in% c("numeric"))
```


We then would like to impute missing values as many exist in our training data.


```r
preModel <- preProcess(training[, num_features_idx], method = c("knnImpute"))
ptraining <- cbind(training$classe, predict(preModel, training[, num_features_idx]))
ptesting <- cbind(testing$classe, predict(preModel, testing[, num_features_idx]))
prtesting <- predict(preModel, raw_testing[, num_features_idx])
names(ptraining)[1] <- "classe"
names(ptesting)[1] <- "classe"
```


## Build Random Forest Model
We build a random forest model from our data to get good prediction accuracy. 

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.0.3
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
rf_model <- randomForest(classe ~ ., ptraining, ntree = 500, mtry = 32)
```


## Perform Cross Validation
This step will help us determine if we have variance due to overfitting.
### In-sample accuracy

```r
training_pred <- predict(rf_model, ptraining)
print(confusionMatrix(training_pred, ptraining$classe))
```

```
## Warning: package 'e1071' was built under R version 3.0.3
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5022    0    0    0    0
##          B    0 3418    0    0    0
##          C    0    0 3080    0    0
##          D    0    0    0 2895    0
##          E    0    0    0    0 3247
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

Table indicates an accuracy of 1.0 and does not suffer from bias.
### Out-of-sample accuracy

```r
testing_pred <- predict(rf_model, ptesting)
```

Confusion Matrix: 

```r
print(confusionMatrix(testing_pred, ptesting$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 556   2   0   0   0
##          B   0 373   0   0   2
##          C   1   3 339   1   0
##          D   0   0   3 319   0
##          E   1   1   0   1 358
## 
## Overall Statistics
##                                         
##                Accuracy : 0.992         
##                  95% CI : (0.987, 0.996)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.99          
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.996    0.984    0.991    0.994    0.994
## Specificity             0.999    0.999    0.997    0.998    0.998
## Pos Pred Value          0.996    0.995    0.985    0.991    0.992
## Neg Pred Value          0.999    0.996    0.998    0.999    0.999
## Prevalence              0.285    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.190    0.173    0.163    0.183
## Detection Prevalence    0.285    0.191    0.176    0.164    0.184
## Balanced Accuracy       0.997    0.991    0.994    0.996    0.996
```

Confusion Matrix table indicates an accuracy greater than 0.99 or 99%.

## Test Prediction Results.

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


## Conclusion
Our model has shown low out of sample and in sample error. We provide a very good prediction of weight lifting style as measured with accelerometers.

