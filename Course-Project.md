Excutive summary
----------------

The goal of your project is to predict the manner in which they did the
exercise. This is the “classe” variable in the training set. You may use
any of the other variables to predict with. You should create a report
describing how you built your model, how you used cross validation, what
you think the expected out of sample error is, and why you made the
choices you did. You will also use your prediction model to predict 20
different test cases.

Exploratory Data analysis
-------------------------

    library(caret)

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    library(rattle)

    ## Loading required package: tibble

    ## Loading required package: bitops

    ## Rattle: A free graphical interface for data science with R.
    ## Version 5.4.0 Copyright (c) 2006-2020 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

    training <- read.csv("pml-training.csv")
    testing <- read.csv("pml-testing.csv")

    dim(training)

    ## [1] 19622   160

The dataset in composed of 19622 observations and 160 variables. If we
explore the dataset, we would notice than many column have NAs and some
with empty values that aren’t NAs, which can’t be used for predicting.
First step would thus be to clean the dataset and keep only the columns
that are useful.

Data Preparation
----------------

### Data Cleaning

Now let’s reduce the size of the dataset as it is quite large and many
variables aren’t useful for our prediction, such as NAs or empty values,
low variability variables and extraneous variables (first 6 columns).

    # Removing NAs
    NAs <- which(colSums(is.na(training)) != 0)
    training <- training[ ,-NAs]

    # Removing covariates with little to no variability in them (it also helps in removing columns with empty values)
    nzv <- nearZeroVar(training)
    training <- training[,-nzv]

    #Removing extraneous variables
    training <- training[,-c(1:6)]

    dim(training)

    ## [1] 19622    53

As we can see we reduced our number of variables from 160 to 53
(including the response variable). This will make model building much
more efficient.

### Data slicing

Now that our dataset is clean, let’s partition the training data into a
train and test set.

    set.seed(1000)
    inTrain <- createDataPartition(training$classe, p=0.7, list=F)
    train <- training[inTrain, ]
    test <- training[-inTrain, ]

Model Building
--------------

In this section, I’ll build 3 different models and select the most
accurate one to predict the testing dataset. The models that I’ll try
are decision trees, random forest and gradient boosting as they are
effective for classification.

### Decision Trees

    modFitRpart <- train(classe ~., method = "rpart", data=train)
    fancyRpartPlot(modFitRpart$finalModel)

![](Course-Project_files/figure-markdown_strict/unnamed-chunk-7-1.png)

    predictRpart <- predict(modFitRpart,newdata=test)
    confusionMatrix(as.factor(test$classe), predictRpart)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1523   27  121    0    3
    ##          B  487  396  256    0    0
    ##          C  474   34  518    0    0
    ##          D  445  165  354    0    0
    ##          E  173  154  291    0  464
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.4929          
    ##                  95% CI : (0.4801, 0.5058)
    ##     No Information Rate : 0.5271          
    ##     P-Value [Acc > NIR] : 1               
    ##                                           
    ##                   Kappa : 0.3366          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.4910  0.51031  0.33636       NA  0.99358
    ## Specificity            0.9457  0.85457  0.88308   0.8362  0.88594
    ## Pos Pred Value         0.9098  0.34767  0.50487       NA  0.42884
    ## Neg Pred Value         0.6250  0.91993  0.78967       NA  0.99938
    ## Prevalence             0.5271  0.13186  0.26168   0.0000  0.07935
    ## Detection Rate         0.2588  0.06729  0.08802   0.0000  0.07884
    ## Detection Prevalence   0.2845  0.19354  0.17434   0.1638  0.18386
    ## Balanced Accuracy      0.7184  0.68244  0.60972       NA  0.93976

### Random Forests

    modFitRf <- train(classe~., method = "rf", data=train)

    predictRf <- predict(modFitRf,newdata=test)
    confusionMatrix(as.factor(test$classe), predictRf)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1672    2    0    0    0
    ##          B   10 1124    5    0    0
    ##          C    0    4 1020    2    0
    ##          D    0    1    7  952    4
    ##          E    0    0    0    2 1080
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9937          
    ##                  95% CI : (0.9913, 0.9956)
    ##     No Information Rate : 0.2858          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.992           
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9941   0.9938   0.9884   0.9958   0.9963
    ## Specificity            0.9995   0.9968   0.9988   0.9976   0.9996
    ## Pos Pred Value         0.9988   0.9868   0.9942   0.9876   0.9982
    ## Neg Pred Value         0.9976   0.9985   0.9975   0.9992   0.9992
    ## Prevalence             0.2858   0.1922   0.1754   0.1624   0.1842
    ## Detection Rate         0.2841   0.1910   0.1733   0.1618   0.1835
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    ## Balanced Accuracy      0.9968   0.9953   0.9936   0.9967   0.9979

### Gradient Boosting

    modFitGbm <- train(classe ~ ., method="gbm",data=train,verbose=FALSE)

    predictGbm <- predict(modFitGbm,newdata=test)
    confusionMatrix(as.factor(test$classe), predictGbm)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1643   20    8    2    1
    ##          B   48 1057   33    1    0
    ##          C    0   31  982   11    2
    ##          D    2    7   30  920    5
    ##          E    1   15    5   11 1050
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9604          
    ##                  95% CI : (0.9551, 0.9652)
    ##     No Information Rate : 0.2879          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9499          
    ##                                           
    ##  Mcnemar's Test P-Value : 1.453e-07       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9699   0.9354   0.9282   0.9735   0.9924
    ## Specificity            0.9926   0.9828   0.9909   0.9911   0.9934
    ## Pos Pred Value         0.9815   0.9280   0.9571   0.9544   0.9704
    ## Neg Pred Value         0.9879   0.9846   0.9844   0.9949   0.9983
    ## Prevalence             0.2879   0.1920   0.1798   0.1606   0.1798
    ## Detection Rate         0.2792   0.1796   0.1669   0.1563   0.1784
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    ## Balanced Accuracy      0.9812   0.9591   0.9595   0.9823   0.9929

Gradient Boosting did pretty well, with 98.78% accuracy, however it’s
less than the random forest model. The

Conclusion
----------

The decision tree method didn’t do a very good job with an accuracy of
49.29%, it didn’t even consider the ‘D’ classe in the prediction. Random
forests did much better than decision tree with 99.69% accuracy, this
model should be good enough for predicting testing data set. Gradient
Boosting on the other hand also performed well with 96.33% accuracy,
however it’s less than the random forest model. Therefore the model
selected for prediction is **random forest**.

Results
=======

    # Making sure the testing dataset has the same variables (columns) for prediction
    cols <- intersect(names(train), names(testing))

    #Predicting results
    result <- predict(modFitRf, testing[, cols])
    result

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
