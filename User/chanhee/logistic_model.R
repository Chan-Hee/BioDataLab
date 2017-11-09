# Data importing
#data<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/DNN_log_10000.csv")
#result<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/CancerResult_log.csv")

data<-read.csv("/home/tjahn/Data/DNN_log_10000.csv")
result<-read.csv("/home/tjahn/Data/CancerResult_log.csv")

library(stringr)
r_name<-as.character(data[,2])
data<-as.data.frame(t(data[,c(-1,-2,-3,-length(data))]))
r_name<-str_replace_all(r_name," /// ","")
colnames(data)<-r_name

data$result<-result[,2]

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
  
  #num_of_attributes = 10
  
  for(i in 1:num_of_attributes){
    glm_model<-glm(result~get(genes[i]),family = binomial(link = logit),data = train)
    
    beta<-abs(glm_model$coefficients[2])
    names(beta)<-genes[i]
    
    betas<-c(betas,beta)
  }
  
  betas <- sort(betas,decreasing = TRUE)  
  betas30<- betas[1:as.integer(0.3*num_of_attributes)]
  
  # Model Fitting
  y<-"result"
  x<-names(betas30)
  flma<-paste(y,paste(x,collapse ="+"),sep = "~")
  
  final_model<-glm(flma,family = binomial(link = logit),data = train)
  model.aic.forward<-step(final_model,direction = "forward")
  
  train_predict<-predict(model.aic.forward,newdata = train,type = "response")
  train_predict<-ifelse(train_predict>0.5,1,0)
  train_tble<-table(train_predict,actual.class = train$result)
  train_accuracy<-(train_tble[1,1]+train_tble[2,2])/sum(train_tble)
  
  test_predict<-predict(model.aic.forward,newdata = test,type = "response")
  test_predict<-ifelse(test_predict>0.5,1,0)
  test_tble<-table(test_predict,actual.class = test$result)
  test_accuracy<-(test_tble[1,1]+test_tble[2,2])/sum(test_tble)
  
  report<-data.frame(test = test_accuracy, train = train_accuracy)
  reports<-rbind(reports,report)

  
  
  }

setwd("/home/tjahn/Git/User/chanhee")
write.csv(report,"stepwids_fivefold_logistic_result.csv")
