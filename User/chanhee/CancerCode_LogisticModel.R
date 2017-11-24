#data<-read.csv("/Users/chanhee/Data_with_CancerCode.csv",row.names = 1, header = TRUE)
data<-read.csv("/home/tjahn/Data/Data_with_CancerCode.csv",row.names = 1, header = TRUE)

#library(stringr)
#c_name<-colnames(data)
#c_name<-str_replace_all(c_name,"\\.","")
#colnames(data)<-c_name
k = 5
#k = 2
list = 1:k
reports=data.frame()
CC_train_reports = data.frame()
CC_test_reports = data.frame()

for(i in 1:k){
  
  
  train<-subset(data, Index %in% list[-i])
  test<-subset(data, Index %in% c(i))
  genes_ <- names(data) %in% c("Result","Index","CancerCode")
  
  num_of_attributes<-dim(data)[2]-3
  genes <- colnames(data[,!genes_])
  betas <- c()
  
  #num_of_attributes=350
  
  for(x in 1:num_of_attributes){
    glm_model<-glm(Result~get(genes[x]),family = binomial(link = logit),data = train)
    
    beta<-abs(glm_model$coefficients[2])
    names(beta)<-genes[x]
    
    betas<-c(betas,beta)
  }
  
  betas <- sort(betas,decreasing = TRUE)  
  
  for(b in 300){
    betas3<- betas[1:b]
    
    # Model Fitting
    fmla <- as.formula(paste("Result~ ", paste(names(betas3), collapse= "+")))
    
    null_model<-glm(Result~1,family = binomial(link = logit),data = train)
    final_model<-glm(fmla,family = binomial(link = logit), data = train)
    model.aic.forward<-step(null_model,scope = list(lower = null_model,upper = final_model),direction = "forward")
    
    train_predict<-predict(model.aic.forward,newdata = train,type = "response")
    train_predict<-ifelse(train_predict>0.5,1,0)
    train$Prediction<-train_predict
    train$Correct<-train$Prediction == train$Result
    
    train_result_with_CancerCode = tapply(train$Correct,train$CancerCode,function(x){sum(x==TRUE)/sum(x)})
    train_result_without = sum(train$Correct==TRUE)/sum(train$Correct)
    
    test_predict<-predict(model.aic.forward,newdata = test,type = "response")
    test_predict<-ifelse(test_predict>0.5,1,0)
    test$Prediction<-test_predict
    test$Correct<-test$Prediction == test$Result
    
    test_result_with_CancerCode = tapply(test$Correct,test$CancerCode,function(x){sum(x==TRUE)/sum(x)})
    test_result_without = sum(test$Correct==TRUE)/sum(test$Correct)
    

    
    report<-data.frame(test = test_result_without, train = train_result_without,genes = b,five_fold = i)
    reports<-rbind(reports,report)
    
    CC_train_report<-data.frame(train_accuracy = train_result_with_CancerCode,genes = b,five_fold = i)
    CC_train_reports<-rbind(CC_train_reports,CC_train_report)
    
    CC_test_report<-data.frame(test_accuracy = test_result_with_CancerCode,genes = b,five_fold = i)
    CC_test_reports<-rbind(CC_test_reports,CC_test_report)
  }
  
}


setwd("/home/tjahn/Git/User/chanhee")
write.csv(reports,"best_stepwise_logistic_result.csv")
write.csv(CC_train_reports,"best_CancerCode_train_stepwise_logistic_result.csv")
write.csv(CC_test_reports,"best_CancerCode_test_stepwise_logistic_result.csv")