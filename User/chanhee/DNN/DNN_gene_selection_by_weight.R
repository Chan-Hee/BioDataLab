# gene95<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/DNN/result_weigthed_sum_gene_95percent_off.csv")
# gene90<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/DNN/result_weigthed_sum_gene_90percent_off.csv")
# gene80<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/DNN/result_weigthed_sum_gene_80percent_off.csv")
# gene70<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/DNN/result_weigthed_sum_gene_70percent_off.csv")
# gene60<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/DNN/result_weigthed_sum_gene_60percent_off.csv")
# gene50<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/DNN/result_weigthed_sum_gene_50percent_off.csv")
# gene40<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/DNN/result_weigthed_sum_gene_40percent_off.csv")
# gene30<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/DNN/result_weigthed_sum_gene_30percent_off.csv")
# gene20<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/DNN/result_weigthed_sum_gene_20percent_off.csv")
# gene10<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/DNN/result_weigthed_sum_gene_10percent_off.csv")
# gene00<-read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/DNN/result_weigthed_sum_gene_00percent_off.csv")
# gene<-data.frame()
# gene<-rbind(gene,gene95)
# gene<-rbind(gene,gene90)
# gene<-rbind(gene,gene80)
# gene<-rbind(gene,gene70)
# gene<-rbind(gene,gene60)
# gene<-rbind(gene,gene50)
# gene<-rbind(gene,gene40)
# gene<-rbind(gene,gene30)
# gene<-rbind(gene,gene20)
# gene<-rbind(gene,gene10)
# gene<-rbind(gene,gene00)

gene<-read.csv("/Users/chanhee/Downloads/Random6000_Random_input_sum_list.csv")
gene<-gene[,2:5]

library(ggplot2)
library(reshape)
library(tidyr)
library(plyr)
library(scales)

data<-gather(gene,key ="Data_Set" ,value="Accuracy",c("Training_Accuracy","Calibration_Accuracy","Test_Accuracy"))
data$Gene_Elimination<-as.factor(data$Gene_Elimination)
data$Data_Set<-factor(data$Data_Set,levels = c("Training_Accuracy","Calibration_Accuracy","Test_Accuracy"))
ggplot(data,aes(x=Gene_Elimination,y=Accuracy))+
  geom_boxplot(aes(fill=Data_Set))+
  theme_bw()+
  labs(titles = "DNN Gene Selection Randomly",y="Accuracy")+
  theme(plot.title = element_text(size=20, face="bold", color="darkgreen", hjust = 0.5),
        axis.text = element_text(size=10),
        legend.title = element_blank(),
        legend.text = element_text(size = 10),
        axis.title = element_text(size = 15),
        axis.text.x = element_text(angle = 0,hjust=1,size = 15))+
        scale_y_continuous(limits=c(0.93,1),oob = rescale_none)

