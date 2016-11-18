####################################################################################################
#
# Practical Machine Learning Course Project - Source Code
#
####################################################################################################

# NOTE: set your working directory and install required packages first!

# Libraries
library(caret)
library(corrplot)
library(e1071)
library(dplyr)
library(rpart)
library(randomForest)

# Load Data
train.raw <- read.csv("pml-training.csv")
test.raw <- read.csv("pml-testing.csv")

# Data Cleaning
## Use only columns that have no missing values in the test set
is.missing <- sapply(test.raw, function (x) any(is.na(x) | x == ""))
pred_names <- names(test.raw)[!is.missing]

## Remove identity and time stamp columns
pred_names_xt <- pred_names[8:59]
pred_names_x <- c("classe", pred_names_xt)      # add "classe" column (for training set only)

## Apply to train and test datasets
x <- train.raw[, pred_names_x]                  # Reduced training set
xt <- test.raw[, pred_names_xt]                 # Reduced test set
dim(x)
dim(xt)

# Check correlations between predictors
correlation <- cor(x[, -1])
highlycorrelated <- findCorrelation(correlation, cutoff = 0.95)
highlycorrelated      # There are 4 highly correlated predictors
corrplot(correlation, method = "circle", order = "FPC", tl.cex = 0.5)

# Explore data
ggplot(x, aes(x = total_accel_belt, fill = classe)) +
      geom_density(alpha = 0.3) +
      labs(title = "Density Plot of Total_Accel_Belt", 
           subtitle = "Grouped by Exercise Type (classe)",
           caption = "Data from http://groupware.les.inf.puc-rio.br/har")

ggplot(x, aes(x = total_accel_arm, fill = classe)) +
      geom_density(alpha = 0.3) +
      labs(title = "Density Plot of Total_Accel_Arm", 
           subtitle = "Grouped by Exercise Type (classe)",
           caption = "Data from http://groupware.les.inf.puc-rio.br/har")

# Split Data into train and validation set (70 / 30)
set.seed(3526)                                                          # Set seed for reproducibility
inTrain <- createDataPartition(x$classe, p = 0.7, list = FALSE)         # Split the data
xtrain <- x[inTrain, ]                                                  # Training Partition - 70%
xval <- x[-inTrain, ]                                                   # Validation Partition - 30%


# Model 1 Training - Decision Tree
fit_dt <- rpart(classe ~ ., data=xtrain, type="class")

# Model 2 Training - Random Forest
fit_rf <- randomForest(classe ~. , data = xtrain, type="class")


# Validation
# Decision Tree
predictions_dt <- predict(fit_dt, xval, type = "class")
confusionMatrix(predictions_dt, xval$classe)

# Random Forest
predictions_rf <- predict(fit_rf, xval, type = "class")
confusionMatrix(predictions_rf, xval$classe)
varImpPlot(fit_rf)


# Predict 20 Test cases

predictions_rf_test <- predict(fit_rf, xt, type = "class")

pml_write_files = function(x){
      n = length(x)
      for(i in 1:n){
            filename = paste0("problem_id_",i,".txt")
            write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
      }
}

pml_write_files(predictions_rf_test)




