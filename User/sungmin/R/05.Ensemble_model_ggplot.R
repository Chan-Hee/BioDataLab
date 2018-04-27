
install.packages('ggplot2')
install.packages('reshape')
install.packages('tidyr')
install.packages('plyr')
install.packages('scales')

library(ggplot2)
library(reshape)
library(tidyr)
library(plyr)
library(scales)

#read model
for(i in c("0","1","2","3","4")){
  assign(paste0("model_",i),read.csv(paste0("D:/biodatalab/2018-1/ensemble_model/",i,"_data_accuracy.csv"),header = T,sep = ","))
  #assign(paste0("model_",i,"$ensemble_name"),rep(i,120))
}
#add a column which is ensemble model name
model_0$ensemble_name <-rep("correlation & mean 0.2",120)
model_1$ensemble_name <-rep("top 2500 variance ",120)
model_2$ensemble_name <-rep("top 2500 diff",120)
model_3$ensemble_name <-rep("Foundation medicine(265)",120)
model_4$ensemble_name <-rep("Foundation medicine(2267)",120)

#make a data.frame for ggplot
gene <- data.frame()
gene <- rbind(gene,model_0)
gene <- rbind(gene,model_1)
gene <- rbind(gene,model_2)
gene <- rbind(gene,model_3)
gene <- rbind(gene,model_4)
gene<-gene[,2:5]

#ggplot
data<-gather(gene,key ="Data_Set" ,value="Accuracy",c("train","val","test"))
data$ensemble_name<-factor(data$ensemble_name,levels = c("correlation & mean 0.2","top 2500 variance ","top 2500 diff","Foundation medicine(265)","Foundation medicine(2267)"
))
data$Data_Set<-factor(data$Data_Set,levels = c("train","val","test"))
ggplot(data,aes(x=ensemble_name,y=Accuracy))+
  geom_boxplot(aes(fill=Data_Set))+
  theme_bw()+
  labs(titles = "Ensemble_model",y="Accuracy")+
  theme(plot.title = element_text(size=20, face="bold", color="darkgreen", hjust = 0.5),
        axis.text = element_text(size=10),
        legend.title = element_blank(),
        legend.text = element_text(size = 10),
        axis.title.y = element_text(size = 15),
        axis.title.x = element_blank(), 
        axis.text.x = element_text(angle = 0,hjust=0.7,size = 7))+
  scale_y_continuous(limits=c(0.93,1),oob = rescale_none)

