# Data importing
#data<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/DNN_log_10000.csv")
#result<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/CancerResult_log.csv")

data<-read.csv("/home/tjahn/Data/DNN_log_10000.csv")
result<-read.csv("/home/tjahn/Data/CancerResult_log.csv")

library(stringr)
r_name<-as.character(data[,2])
data<-as.data.frame(t(data[,c(-1,-2,-3,-length(data))]))
r_name<-str_replace_all(r_name," /// ","")
r_name<-str_replace_all(r_name,"/","")

colnames(data)<-r_name

data$Result<-result[,2]

# Random Sampling -> Test & Training Data set
num_of_samples <- dim(data)[1]

##rand_index <- sample(1:num_of_samples,num_of_samples)
#test_index <- rand_index[1:as.integer(num_of_samples*0.2)]
##train_index <- rand_index[as.integer(num_of_samples*0.2+1):num_of_samples]
##train<-data[train_index,]
##test<-data[test_index,]

# 5-fold cross validation
k = 5
data$id<-sample(1:k,nrow(data),replace = TRUE)
list = 1:k
reports=data.frame()
for(i in 1:k){
  train<-subset(data, id %in% list[-i])
  test<-subset(data, id %in% c(i))
  results <- names(data) %in% "result"
  
  num_of_attributes<-dim(data)[2]-2
  genes <- colnames(data[,!results])
  betas <- c()
  
  #num_of_attributes = 100
  
  for(i in 1:num_of_attributes){
    glm_model<-glm(Result~get(genes[i]),family = binomial(link = logit),data = train)
    
    beta<-abs(glm_model$coefficients[2])
    names(beta)<-genes[i]
    
    betas<-c(betas,beta)
  }
  
  betas <- sort(betas,decreasing = TRUE)  
  betas3<- betas[1:as.integer(0.03*num_of_attributes)]
  
  # Model Fitting

  x<-names(betas3)
  temp1<-train$Result
  temp2<-test$Result
  
  train<-train[,x]
  train$Result<-temp1
  test<-test[,x]
  test$Result<-temp2
  null_model<-glm(Result~1,family = binomial(link = logit),data = train)
  final_model<-glm(Result ~ .,family = binomial(link = logit), data = train)
  model.aic.forward<-step(null_model,scope = list(lower = null_model,upper = final_model),direction = "forward")
  
  train_predict<-predict(model.aic.forward,newdata = train,type = "response")
  train_predict<-ifelse(train_predict>0.5,1,0)
  train_accuracy<-sum(train_predict==train$Result)/nrow(train)
    
  test_predict<-predict(model.aic.forward,newdata = test,type = "response")
  test_predict<-ifelse(test_predict>0.5,1,0)
  test_accuracy<-sum(test_predict==test$Result)/nrow(test)
  
  report<-data.frame(test = test_accuracy, train = train_accuracy)
  reports<-rbind(reports,report)

  
  
  }

setwd("/home/tjahn/Git/User/chanhee")
write.csv(reports,"stepwise_fivefold_logistic_result.csv")
