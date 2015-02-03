# Practical Machine Learning - Course Project
Eric Olson  
Saturday, January 24, 2015  


#### Executive Summary  
Using datasets provided from personal fitness devices, a machine learning model is developed to predict in which of 5 manners a participant completed an exercise incorrectly.  That is, predicting which class a sample belongs to -- A, B, C, D, or E.  The datasets were cleaned and the total number of predicting features reduced from 160 to 52 in the final model. The training data was split 70/30 into training and testing sets. A random forest model with oob resampling was chosen and worked very well. The final model produced 100% accuracy for the final validation cases (20 total).  The estimated OOB error rate was 0.88% (99.12% accuracy on the training data set).  The measured accuracy for the testing subset (cross-validation set) was 99.4%. Reducing ntree from the default of 500 to 100 had very little effect on the accuracy of the model or OOB error estimate, but reduced the run time by a factor of about 6.  


#### Introduction
There is a growing number of people wearing personal fitness devices to track physical activities. The devices typcially quantify how much of a particular activity is done, but they rarely quantify how well it is done. In this report, data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants are used. Participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. This data is then used to build models to classify and then predict in which of the 5 ways the participant incorrectly performed the activity. More information about the data is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).  

#### Approach  
Before building the model, the data set was examined to determine what it contained and what tidying was needed.  


```r
require(caret) # Used in building model later
# Load the data and take a look
traindata <- read.csv("./pml-training.csv", stringsAsFactors=FALSE)
validation <- read.csv("./pml-testing.csv", stringsAsFactors=FALSE)  
```


```r
dim(traindata)
```

```
## [1] 19622   160
```

Several steps were needed to prepare the training data for modeling:  

1. Summary rows for each window were removed (new_window = yes), since they are not contained in the final test set, and cannot be included in the prediction model.  
2. Columns that have no values (missing or NA, after step 1 is completed above) were removed.  
3. Columns such as date, time, user name, which imply a structure to the data that is arbitrary and, therefore, unsuitable for a generalized prediction model, were also removed (the first 7 columns)  
4. The classe is class "chr", convert to a factor variable for use with random forest model.


```r
traindata <- traindata[which(traindata$new_window=="no"),]
traindata[traindata==""] <- NA
traindata <- traindata[colSums(is.na(traindata))==0]
traindata <- traindata[, 8:ncol(traindata)]
traindata$classe <- as.factor(traindata$classe)
dim(traindata)
```

```
## [1] 19216    53
```

The features (columns) have been reduced from 160 to 53 (52 predictors + 1 outcome (classe)).  For the final validation cases (20), the same columns as above were retained, with the exception that the problem_id column was removed.  

```r
keep <- colnames(traindata[1:ncol(traindata)-1]) # last column is problem_id
validation <- validation[,keep]
```

With the data in a tidy form, the training data was split into 2 sets: training (70%) / testing (30%), which provides a way of cross-validating the model and calculating out-of-sample (OOB) error / accuracy before predicting on the final validation set.  Also, the random seed was set for reproducibility.  


```r
set.seed(98102)
inTrain = createDataPartition(traindata$classe, p = 0.7, list=FALSE)
training = traindata[inTrain,]
testing = traindata[-inTrain,]  # note, the original "testing" data was renamed "validation" (20 cases)
```

##### Model-Building
Based on class discussions and the great performance of Random Forest models, especially for non-linear data, this was chosen as a starting point.  Initally, the default of ntree = 500 was used.  The resampling method (used to estimate OOB error rate) was set to "oob", which is recommended for random forest models.  If the training set were not split into training/testing sets, "cv" resampling with number = 3 (k-folds) might be more appropriate.  See <http://topepo.github.io/caret/training.html#custom>  


```r
trOpt <- trainControl(method="oob") # out-of-bag error estimate
rfModl <- train(training$classe ~ . , method = "rf", trControl = trOpt, data = training) # ntree=500 (default)
rfPred <- predict(rfModl, testing) # predict on cross-validation data
rfCM <- confusionMatrix(testing$classe, rfPred) # check accuracy 
rfVal<- predict(rfModl, validation) # Validation set (20 cases)
```

Refine model to reduce ntree to 100 (reduce time to run by factor of 6), and calculate same measures.

```r
set.seed(98102)
trOpt2 <- trainControl(method="oob")
rfModl2 <- train(training$classe ~ . , method = "rf", ntree = 100, trControl = trOpt2, data = training)
rfPred2 <- predict(rfModl2, testing) # predict on cross-validation data
rfCM2 <- confusionMatrix(testing$classe, rfPred2)
rfVal2<- predict(rfModl2, validation)  # Validation set (20 cases)
```

#### Results

Model 1: 
Looking at the first model from above (rfModl), we can see that the out-of-sample error estimate (OOB estimate of error rate) is 0.81% (99+% accuracy), and mtry=2 was chosen.  Run time was 3 - 4 minutes (Intel Core i5, 8 GB RAM, Windows 8.1)


```r
rfModl$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.81%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3829    0    1    0    0 0.0002610966
## B   17 2579    7    0    0 0.0092201306
## C    0   27 2316    4    0 0.0132083511
## D    0    0   42 2156    5 0.0213345438
## E    0    0    0    6 2464 0.0024291498
```

The measured out-of-sample accuracy on the cross-validation set was 99.5% as shown by the confusion matrix.  Accuracy on the final validation set (20 cases) was 100% (not shown).  


```r
rfCM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1641    0    0    0    0
##          B    8 1106    1    0    0
##          C    0    4 1000    1    0
##          D    0    0   13  931    0
##          E    0    0    0    2 1056
## 
## Overall Statistics
##                                           
##                Accuracy : 0.995           
##                  95% CI : (0.9928, 0.9966)
##     No Information Rate : 0.2861          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9936          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9951   0.9964   0.9862   0.9968   1.0000
## Specificity            1.0000   0.9981   0.9989   0.9973   0.9996
## Pos Pred Value         1.0000   0.9919   0.9950   0.9862   0.9981
## Neg Pred Value         0.9981   0.9991   0.9971   0.9994   1.0000
## Prevalence             0.2861   0.1926   0.1760   0.1621   0.1832
## Detection Rate         0.2847   0.1919   0.1735   0.1615   0.1832
## Detection Prevalence   0.2847   0.1935   0.1744   0.1638   0.1836
## Balanced Accuracy      0.9976   0.9972   0.9926   0.9970   0.9998
```

Model 2: 
The second model simply reduces ntree to 100, which reduces the run time to ~ 30 - 40 seconds (same machine and set-up as for Model 1).  Out-of-sample error estimate (OOB estimate of error rate) is 0.88% (99+% accuracy), but mtry=27, meaning a different sampling at each split was chosen as the final model (based on accuracy).  


```r
rfModl2$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, ntree = 100, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 100
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.88%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3825    2    3    0    0 0.001305483
## B   18 2572   10    2    1 0.011909335
## C    0   16 2316   15    0 0.013208351
## D    0    1   30 2167    5 0.016341353
## E    0    3    4    8 2455 0.006072874
```

The measured out-of-sample accuracy on the cross-validation set was 99.46% as shown by the confusion matrix.  Accuracy on the final validation set (20 cases) was 100% (not shown).  


```r
rfCM2
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1640    0    0    0    1
##          B    8 1104    3    0    0
##          C    0    5  998    2    0
##          D    0    0   10  934    0
##          E    0    1    0    4 1053
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9941          
##                  95% CI : (0.9918, 0.9959)
##     No Information Rate : 0.286           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9925          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9951   0.9946   0.9871   0.9936   0.9991
## Specificity            0.9998   0.9976   0.9985   0.9979   0.9989
## Pos Pred Value         0.9994   0.9901   0.9930   0.9894   0.9953
## Neg Pred Value         0.9981   0.9987   0.9973   0.9988   0.9998
## Prevalence             0.2860   0.1926   0.1754   0.1631   0.1829
## Detection Rate         0.2846   0.1916   0.1732   0.1621   0.1827
## Detection Prevalence   0.2847   0.1935   0.1744   0.1638   0.1836
## Balanced Accuracy      0.9975   0.9961   0.9928   0.9958   0.9990
```


#### Final Thoughts
The random forest model was my initial model choice.  Since it worked so well, and was quick to execute with the options chosen, it became my final model.  I tried other models such as using k-fold options with "cv" resampling, which required 5 - 10 min to run, but accuracy was lower than random forests.  Another option would be to try gbm or doing pca followed by lda, ls, or other linear method.  If rf had not been so quick and easy, I would have further pursued these other options.





#### 
