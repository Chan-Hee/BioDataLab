d_test1<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/dnn_acuracy_by_cancer_test1.csv")
d_test2<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/dnn_acuracy_by_cancer_test2.csv")
d_test3<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/dnn_acuracy_by_cancer_test3.csv")
d_test4<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/dnn_acuracy_by_cancer_test4.csv")
d_test5<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/dnn_acuracy_by_cancer_test5.csv")

d_train1<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/dnn_acuracy_by_cancer_train1.csv")
d_train2<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/dnn_acuracy_by_cancer_train2.csv")
d_train3<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/dnn_acuracy_by_cancer_train3.csv")
d_train4<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/dnn_acuracy_by_cancer_train4.csv")
d_train5<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/dnn_acuracy_by_cancer_train5.csv")

l_test1<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/logistic_accuracy_by_cancer_test1.csv")
l_test2<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/logistic_accuracy_by_cancer_test2.csv")
l_test3<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/logistic_accuracy_by_cancer_test3.csv")
l_test4<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/logistic_accuracy_by_cancer_test4.csv")
l_test5<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/logistic_accuracy_by_cancer_test5.csv")

l_train1<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/logistic_accuracy_by_cancer_train1.csv")
l_train2<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/logistic_accuracy_by_cancer_train2.csv")
l_train3<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/logistic_accuracy_by_cancer_train3.csv")
l_train4<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/logistic_accuracy_by_cancer_train4.csv")
l_train5<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/Accuracy/input/logistic_accuracy_by_cancer_train5.csv")

calculate_acc<-function(x){
  sum<-sum(x$number_of_patients)
  tot_acc<-sum(x$accuracy*x$number_of_patients)
  tot_acc<-tot_acc/sum
  return(tot_acc)
}

dnn_test<-list(d_test1,d_test2,d_test3,d_test4,d_test5)
dnn_train<-list(d_train1,d_train2,d_train3,d_train4,d_train5)
log_test<-list(l_test1,l_test2,l_test3,l_test4,l_test5)
log_train<-list(l_train1,l_train2,l_train3,l_train4,l_train5)

d_test_result<-c()
for(i in dnn_test){
  d_test_result<-c(d_test_result,calculate_acc(i))
}

d_train_result<-c()
for(i in dnn_train){
  d_train_result<-c(d_train_result,calculate_acc(i))
}

l_test_result<-c()
for(i in log_test){
  l_test_result<-c(l_test_result,calculate_acc(i))
}

l_train_result<-c()
for(i in log_train){
  l_train_result<-c(l_train_result,calculate_acc(i))
}

library(reshape)
library(ggplot2)
library(tidyr)
library(plyr)
library(scales)
df<-data.frame(dnn_test_accuracy = d_test_result,dnn_train_accuracy = d_train_result,logistic_test_accuracy = l_test_result,logistic_train_accuracy = l_train_result)
melt_df<-melt(df)
p_df<-separate(melt_df,variable,into = c("model","group"),sep ="_",extra = "drop")
p_df2<-ddply(p_df,c("model","group"),summarise,mean=mean(value),sd=sd(value),n=length(value),se=sd/sqrt(n))

p_df2$group<-factor(p_df2$group,levels = c("train","test"))
ggplot(p_df2,aes(x=model,y=mean,fill=group))+
  geom_bar(position = position_dodge(),stat = "identity")+
  scale_y_continuous(limits=c(0.8,1),oob = rescale_none)+
  geom_errorbar(aes(ymin=mean-sd, ymax=mean+sd),
                width=.2,                    # Width of the error bars
                position=position_dodge(.9))+
  labs(titles = "Best Model Comparison",x="",y="Accuracy")+
  theme_bw()+
  theme(plot.title = element_text(size=20, face="bold", color="darkgreen", hjust = 0.5),
        axis.text = element_text(size=10),
        legend.title = element_blank(),
        legend.text = element_text(size = 15),
        axis.title = element_text(size = 15),
        axis.text.x = element_text(angle = 0,hjust=1,size = 15))
  