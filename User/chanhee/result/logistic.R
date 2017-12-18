library(reshape)
library(ggplot2)

data<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/result/final_stepwise_fivefold_logistic_result.csv",header = T)
data<-data[,-4]
melt_data<-melt(data,id.var ="genes")
melt_data$variable<-factor(melt_data$variable,levels = c("train","test"))
ggplot(melt_data,aes(x = factor(genes), y = value,fill =variable))+
  geom_boxplot()+
  labs(titles="Logistic Model Accuracy",x ="Number of Genes", y = "Accuracy")+
  theme_bw()+
  theme(plot.title = element_text(size=20, face="bold", color="darkgreen", hjust = 0.5),
        axis.text = element_text(size=15),
        legend.title = element_blank(),
        legend.text = element_text(size = 15),
        axis.title = element_text(size = 15))
