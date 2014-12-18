# PML Project
Vincent Rupp  
Tuesday, December 09, 2014  
###Background and Introduction  
Six people did 10 reps of bicep curls five different ways while wearing sensors. One of the ways was with correct form; the others were incorrect. Our goal is to predict the form by using the sensor data.  

I'll create a random forest model on a random subset of the training set, estimate its accuracy using cross-validation, and then create a final model using the entire data set to be applied to the 20 test cases that this project grades on.  

###Analysis  
Set up the properties and libraries for this document.  

```r
library(knitr)
library(lattice)
library(ggplot2)
library(caret)
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
setwd("C:/Coursera/Data_Science/08_Practical_Machine_Learning/PML_CourseProject")
set.seed(5)
opts_chunk$set(echo=TRUE,results="show",fig.align="center",cache=TRUE)
```

Import the data.  

```r
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
dim(training); dim(testing)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```

It appears from some preliminary analysis that there are a lot of columns and some rows I need to get rid of:  
1. 67 of the columns have 19216/19622 NAs. No other columns have any NA values.  
2. 24 of the columns start with "kurtosis" or "skewness" and are blank except on lines where the new_window column equals "yes". Further, some non-blank values are "#DIV/0!" Therefore, all kurtosis and skewness columns, along with new_window will be removed.  
3. There's a similar pattern with min_yaw, max_yaw, and amplitude_yaw, so those are also gone.  
(Note: The new_window=='yes' rows are where the non-NA for mostly-NA columns are as well.)  
4. I don't think the timestamps will be very useful to model either.  
5. And I don't know what num_window is either, but I don't like it.  
6. X is the row number and is completely related to classe, so it has to go.


```r
#First I'll make a vector of the new_window=='yes' rows
new_windows <- training$new_window=='yes'

#Make a vector of TRUE values - the default is to keep each row
keep <- rep(TRUE,dim(training)[2])

#Set keep to FALSE when any of the column conditions are met
#(There's more efficient code to achieve the result, but this has the benefit of being really clear.)
for (i in 1:length(names(training))) {
  if (sum(is.na(training[,i])) > 5000) {keep[i] <- FALSE}
  if (grepl("^kurtosis|skewness",names(training)[i])==TRUE) {keep[i] <- FALSE}
  if (grepl("^min_yaw|max_yaw|amplitude_yaw",names(training)[i])==TRUE) {keep[i] <- FALSE}
  if (grepl("timestamp",names(training)[i])==TRUE) {keep[i] <- FALSE}
  if (names(training)[i] == "new_window") {keep[i] <- FALSE}
  if (names(training)[i] == "num_window") {keep[i] <- FALSE}
  if (names(training)[i] == "X") {keep[i] <- FALSE}
}

trainingUse <- training[!new_windows,keep]
```

Now I'll take a random subset of just 2000 observations to set up a preliminary model. That way, I limit runtime and make sure I know what I'm doing.  


```r
sampleObs <- sample(1:dim(trainingUse)[1],2000,replace=FALSE)
trainingUseSub <- trainingUse[sampleObs,]
```

Now to actually build a model. I'll use a random forest by way of the train() function in the caret package.  

I'll use trainControl to use a 5-fold cross-validation. I don't have any great reason for that, just that k-fold cross-validation should be quick and 5 seems like a reasonable number for this exploratory set.  

```r
modFit <- train(classe ~ .,data=trainingUseSub,method="rf",
                trControl=trainControl(method="cv",number=5))
print(modFit)
```

```
## Random Forest 
## 
## 2000 samples
##   53 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 1599, 1600, 1599, 1601, 1601 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD  
##    2    0.9249884  0.9048524  0.009131772  0.01161952
##   29    0.9384947  0.9220652  0.009506901  0.01208070
##   57    0.9299959  0.9113261  0.009892207  0.01254247
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 29.
```
The caret package optimized mtry for me, which I find touching. The achieved accuracy on this 2000-row set is almost 94%. Not bad at all.  

Let's check its predictions on the entire training set of over 19000 observations.  


```r
confusionMatrix(training$classe,predict(modFit,training))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5522   21   18    2   17
##          B  259 3401   88   22   27
##          C    9  101 3228   64   20
##          D   29   20   76 3076   15
##          E    7   15   43   33 3509
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9548          
##                  95% CI : (0.9518, 0.9577)
##     No Information Rate : 0.2969          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9428          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9478   0.9559   0.9348   0.9622   0.9780
## Specificity            0.9958   0.9753   0.9880   0.9915   0.9939
## Pos Pred Value         0.9896   0.8957   0.9433   0.9565   0.9728
## Neg Pred Value         0.9784   0.9901   0.9861   0.9926   0.9951
## Prevalence             0.2969   0.1813   0.1760   0.1629   0.1829
## Detection Rate         0.2814   0.1733   0.1645   0.1568   0.1788
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9718   0.9656   0.9614   0.9768   0.9859
```
Accuracy is very similar to what was achieved on the 2000-row set. We expect the out-of-sample accuracy to be similar, assuming the test set was chosen randomly from the same population and this model doesn't overfit to the training set.   

Now that I feel good about the method, I'll build a model based on the entire 19,216-row training set. Instead of 5-fold cross-validation, I'll use the bootstrap method and 25 iterations. (25 is the default, so I assume there's some research showing that's an optimal number.) 


```r
modFitFinal <- train(classe ~ .,data=trainingUse,method="rf",
                trControl=trainControl(method="boot",number=25))
```

That takes almost three hours to run, so here's hoping it's worth it!  


```r
confusionMatrix(training$classe,predict(modFitFinal,training))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5580    0    0    0    0
##          B    0 3796    1    0    0
##          C    0    0 3422    0    0
##          D    0    0    1 3215    0
##          E    0    0    0    0 3607
## 
## Overall Statistics
##                                      
##                Accuracy : 0.9999     
##                  95% CI : (0.9996, 1)
##     No Information Rate : 0.2844     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 0.9999     
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   0.9994   1.0000   1.0000
## Specificity            1.0000   0.9999   1.0000   0.9999   1.0000
## Pos Pred Value         1.0000   0.9997   1.0000   0.9997   1.0000
## Neg Pred Value         1.0000   1.0000   0.9999   1.0000   1.0000
## Prevalence             0.2844   0.1935   0.1745   0.1638   0.1838
## Detection Rate         0.2844   0.1935   0.1744   0.1638   0.1838
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   0.9997   1.0000   1.0000
```

The accuracy is now 99.99%. There was only one classe miscategorized out of 19216 observations.  

Let's hope I did this all correctly so that the testing set matches the expected answers when I submit them all.  

I'll apply modFitFinal to the testing data set and then use the suggested method to output 20 text files containing a single character each.  


```r
answers <- predict(modFitFinal,testing)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
setwd("C:/Coursera/Data_Science/08_Practical_Machine_Learning/project_answers")
pml_write_files(as.vector(answers))
```

#WHEW!  

All 20 were correct. [sunglasses-emoji]
