#load library
library(C50)

#load data files
trainingData <- read.csv("data.csv", header=FALSE, sep=",")
trainingLabel <- read.csv("data_labels.csv", header=FALSE, sep=",")
testingData <- read.csv("test_set.csv", header=FALSE, sep=",")

#Append trainingLabel to trainingData
trainingData["label"] <- trainingLabel

#Split data into training set and testing set 70:30
validationSet <- trainingData[2441:3486,]
trainingSet <- trainingData[1:2440,]

#assign column names to data files
trainingSet$label<-as.factor(trainingSet$label)

set.seed(1) #makes sure results are reproducible

ptm <- proc.time()

#Initial Model: build tree model using C5.0 algorithm
#treeModel <- C5.0(label ~ ., data = trainingSet, cost=finalCost) 
#Equivalent to treeModel <- C5.0(x = trainingSet, y = trainingLabel)
#pred <- predict(treeModel, testingSet)

#Define Cost Matrix
finalCost <- matrix(
  c(0, 0, 0, 0, 0, 0, 0, 0,
    50, 0, 50, 50, 50, 50, 50, 50,
    100, 100, 0, 100, 100, 100, 100, 100,
    0, 0, 0, 0, 0, 0, 0, 0,
    20, 20, 20, 20, 0, 20, 20, 20,
    10, 10, 10, 10, 10, 0, 10, 10,
    90, 90, 90, 90, 90, 90, 0, 90,
    0, 0, 0, 0, 0, 0, 0, 0), 
  nrow=8, ncol = 8)

#boosting aids to increase accuracy of the tree model by adding weak learners 
#such that new learners pick up the slack of old learners
#Reference: http://connor-johnson.com/2014/08/29/decision-trees-in-r-using-the-c50-package/
boostTreeModel <- C5.0(label ~ ., data = trainingSet, trials = 5, cost=finalCost)
pred1 <- predict(boostTreeModel, validationSet)
acc = sum(pred1 == validationSet$label) / length(pred1)

#Predict the test-set.csv (unknown label)
pred <- predict(boostTreeModel, testingData)
summary(boostTreeModel)
proc.time() - ptm

#sink(file = "predicted_other.csv")
write.table(pred, file = "predicted_other.csv", sep = ",", col.names = NA,qmethod = "double")