# Function Definition #
GetAccuracy<-function(x){
  return(c(accuracy = sum(x$prediction==x$result)/nrow(x), number_of_patients =  nrow(x)))
}


# Data Imoprting #
sample_info<-read.csv("/Users/chanhee/Desktop/GPL570_sampleinfo.txt",sep = "\t")
data<-read.csv("/Users/chanhee/Desktop/BioDataLab/Data/output/result_file_test4.csv",row.names = 1,header = T)



# Adding CancerCode to Original Data #
row.names(sample_info)<-sample_info$GSM_ID
GSM<-row.names(data)
sample_info<-sample_info[GSM,]
data$cancer_code<-as.factor(as.character(sample_info$CANCER_CODE))


# Drawing Accuracy Plot #
#plot_data<- data[,c("result","prediction","probability","cancer_code")]
plot_data<-data

result<-by(plot_data,plot_data$cancer_code,GetAccuracy)
result_df<-as.data.frame(t(as.data.frame.list(result)))
result_df$accuracy<-round(result_df$accuracy,3)
result_df<-result_df[rev(order(result_df$number_of_patients)),]


write.csv(result_df,"/Users/chanhee/Desktop/DNN_accuracy_CancerCode/dnn_acuracy_by_cancer_test5.csv")

