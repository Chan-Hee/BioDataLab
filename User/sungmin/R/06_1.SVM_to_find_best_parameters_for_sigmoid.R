library(e1071)
library(rpart)
library(caret)
data <- read.csv("/home/tjahn/Data/input_ensemble/selected_model_0_data.csv", sep = ",", header = T)
data$result <-as.factor(data$result)

test <-data[data$index == 0,]
train <-data[data$index != 0, ]
test <- subset(test, select = -c(index,patient,cancer_code))  
train <- subset(train, select = -c(index,patient,cancer_code))

data <- subset(data, select = -c(index,patient,cancer_code))

svm_train<-tune.svm(result~.,data = train,kernel = 'sigmoid',gamma = c(0.1,0.3,0.5,0.7,1,1.5,2,3,5),coef0 = c(0.1,0.3,0.5,0.7,1,1.5,2,3,5),cost = c(0.001,0.005,0.01,0.05,0.1,0.3,0.5,1,1.5,2,5,10))
svm_data<-tune.svm(result~.,data = data,kernel = 'sigmoid',gamma = c(0.1,0.3,0.5,0.7,1,1.5,2,3,5),coef0 = c(0.1,0.3,0.5,0.7,1,1.5,2,3,5),cost = c(0.001,0.005,0.01,0.05,0.1,0.3,0.5,1,1.5,2,5,10)

                   
write(svm_train$best.parameters,"/home/tjahn/tf_save_data/sungmin/svm_train.csv")
write(svm_data$best.parameters,"/home/tjahn/tf_save_data/sungmin/svm_data.csv")
                   