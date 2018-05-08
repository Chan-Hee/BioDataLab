//install.packages("e1071")
//install.packages("rpart")
//install.packages("caret")
library(e1071)
library(rpart)
library(caret)


#Data
for(i in 0:4){
  result <- data.frame()
  data <- read.csv(paste0("/home/tjahn/Data/input_ensemble/selected_model_",i,"_data.csv"), sep = ",", header = T)
  data$result <-as.factor(data$result)
  for(j in 1:5){
    test <-data[data$index == j,]
    train <-data[data$index != j, ]
    test <- sebset(test, select = -c(index,patient,cancer_code))  
    train <- sebset(train, select = -c(index,patient,cancer_code))
    
    svm_model <- svm(result~.,data = train, kernel = "sigmoid", cost = 0.1, gamma= 0.45,coef.0 = 0 ,epsilon = 0.1)
    pred <- predict(svm_model,test)
    result_table<- table(pred,test$result)
    auc <- sum(result_table[1,1],result_table[2,2])/sum(result_table)
    result[j,1] = auc
    result[j,2] = j
  }
  write.csv(result,paste0("/home/tjahn/tf_save_data/sungmin/result/SVM/result_",i,".csv"))
}
