data<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/result/final_stepwise_fivefold_logistic_result.csv")
data<-data[,-1]

train<-tapply(data$train,data$genes,mean)
test<-tapply(data$test,data$genes,mean)

plot(y=train,x = names(train),xlab = "Genes Number",ylim = c(0.8,1),type ="b",ylab = "Accuracy",
     ,main = "Cancer Prediction")
points(y=test,x = names(test),xlab = "Genes Number",ylim = c(0.8,1),col="red",type="b")

legend("topright",legend = c("train_accuracy", "test_accuracy"),pch =c(1,1),col = c("black","red"))
