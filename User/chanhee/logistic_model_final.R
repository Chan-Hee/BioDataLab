# Data importing


data<-read.csv("/home/tjahn/Data/FinalData_GSM_gene_index_result.csv.csv",row.names = 1)
#data<-read.csv("/Users/chanhee/FinalData_GSM_gene_index_result.csv",row.names = 1)
setwd("/home/tjahn/Git/User/chanhee")





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
    
    #num_of_attributes = 100 #test
    
    for(x in 1:num_of_attributes){
      glm_model<-glm(result~get(genes[x]),family = binomial(link = logit),data = train)
      
      beta<-abs(glm_model$coefficients[2])
      names(beta)<-genes[x]
      
      betas<-c(betas,beta)
    }
    
    betas <- sort(betas,decreasing = TRUE)  
    
    b = 300 #test
    betas3<- betas[1:b]
    select_genes<-names(betas3)
    
    train<-train[,c(select_genes,"result")]
    test<-test[,c(select_genes,"result")]
    # Model Fitting
    fmla <- as.formula(paste("result~ ", paste(names(betas3), collapse= "+")))
    
    null_model<-glm(result~1,family = binomial(link = logit),data = train)
    final_model<-glm(fmla,family = binomial(link = logit), data = train)
    model.aic.forward<-step(null_model,scope = list(lower = null_model,upper = final_model),direction = "forward")
    
    train_predict<-predict(model.aic.forward,newdata = train,type = "response")
   
    
    prediction<-ifelse(train_predict>0.5,1,0)
    train$prediction<-prediction
    train$probability<-train_predict
    
    train_accuracy<-sum(train$prediction==train$result)/nrow(train)
    
    test_predict<-predict(model.aic.forward,newdata = test,type = "response")
    
    
    prediction<-ifelse(test_predict>0.5,1,0)
    test$prediction<-prediction
    test$probability<-test_predict
    test_accuracy<-sum(test$prediction==test$result)/nrow(test)
    
    report<-data.frame(test = test_accuracy, train = train_accuracy,genes = b,five_fold = i)
    reports<-rbind(reports,report)
    
    if(i==1){
      write.csv(train,"logistic_train_prediction_index1.csv")
      write.csv(test,"logistic_test_prediction_index1.csv")
    }
    if(i==2){
      write.csv(train,"logistic_train_prediction_index2.csv")
      write.csv(test,"logistic_test_prediction_index2.csv")
    }
    if(i==3){
      write.csv(train,"logistic_train_prediction_index3.csv")
      write.csv(test,"logistic_test_prediction_index3.csv")
    }
    if(i==4){
      write.csv(train,"logistic_train_prediction_index4.csv")
      write.csv(test,"logistic_test_prediction_index4.csv")
    }
    if(i==5){
      write.csv(train,"logistic_train_prediction_index5.csv")
      write.csv(test,"logistic_test_prediction_index5.csv")
    }

}


 
write.csv(reports,"logistic_result_accuracy.csv")

