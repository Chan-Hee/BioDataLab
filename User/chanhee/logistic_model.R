# Data importing


data<-read.csv("/home/tjahn/Data/FinalData.csv")
#data<-read.csv("FinalData.csv")

# Random Sampling -> Test & Training Data set


# 5-fold cross validation
k = 5

list = 1:k
reports=data.frame()
for(i in 1:k){


    train<-subset(data, index %in% list[-i])
    test<-subset(data, index %in% c(i))
    genes_ <- names(data) %in% c("result","index")
    
    num_of_attributes<-dim(data)[2]-2
    genes <- colnames(data[,!genes_])
    betas <- c()
    
    #num_of_attributes = 100
    
    for(x in 1:num_of_attributes){
      glm_model<-glm(result~get(genes[x]),family = binomial(link = logit),data = train)
      
      beta<-abs(glm_model$coefficients[2])
      names(beta)<-genes[x]
      
      betas<-c(betas,beta)
    }
    
    betas <- sort(betas,decreasing = TRUE)  
    
    for(b in c(50,100,150,200,250,300)){
      betas3<- betas[1:b]
      
      # Model Fitting
      fmla <- as.formula(paste("result~ ", paste(names(betas3), collapse= "+")))
      
      null_model<-glm(result~1,family = binomial(link = logit),data = train)
      final_model<-glm(fmla,family = binomial(link = logit), data = train)
      model.aic.forward<-step(null_model,scope = list(lower = null_model,upper = final_model),direction = "forward")
      
      train_predict<-predict(model.aic.forward,newdata = train,type = "response")
      train_predict<-ifelse(train_predict>0.5,1,0)
      train_accuracy<-sum(train_predict==train$result)/nrow(train)
      
      test_predict<-predict(model.aic.forward,newdata = test,type = "response")
      test_predict<-ifelse(test_predict>0.5,1,0)
      test_accuracy<-sum(test_predict==test$result)/nrow(test)
      
      report<-data.frame(test = test_accuracy, train = train_accuracy,genes = b,five_fold = i)
      reports<-rbind(reports,report)
    }

}

setwd("/home/tjahn/Git/User/chanhee")
write.csv(reports,"final_stepwise_fivefold_logistic_result.csv")
