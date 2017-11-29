
GetAccuracy<-function(x){
  return(c(accuracy = sum(x$prediction==x$result)/nrow(x), number_of_patients =  nrow(x)))
}



sample_info<-read.csv("/Users/chanhee/Desktop/GPL570_sampleinfo.txt",sep = "\t")
data<-read.csv("/Users/chanhee/Desktop/plot_input_logistic/logistic_train_prediction_index5.csv",row.names = 1,header = T)




row.names(sample_info)<-sample_info$GSM_ID
GSM<-row.names(data)
sample_info<-sample_info[GSM,]
data$cancer_code<-as.factor(as.character(sample_info$CANCER_CODE))



plot_data<- data[,c("result","prediction","probability","cancer_code")]


result<-by(plot_data,plot_data$cancer_code,GetAccuracy)
result_df<-as.data.frame(t(as.data.frame.list(result)))
result_df$accuracy<-round(result_df$accuracy,3)
result_df<-result_df[rev(order(result_df$number_of_patients)),]

result_df5<-result_df
dfs<-list(row.names(result_df1),row.names(result_df2),row.names(result_df3),row.names(result_df4),row.names(result_df5))
common_name<-Reduce(intersect,dfs)


result_df1<-result_df1[common_name,]
result_df2<-result_df2[common_name,]
result_df3<-result_df3[common_name,]
result_df4<-result_df4[common_name,]
result_df5<-result_df5[common_name,]

accuracy<-data.frame(df1=result_df1$accuracy,df2=result_df2$accuracy,df3=result_df3$accuracy,df4=result_df4$accuracy,df5=result_df5$accuracy)
number_of_patients<-data.frame(df1=result_df1$number_of_patients,df2=result_df2$number_of_patients,df3=result_df3$number_of_patients,df4=result_df4$number_of_patients,df5=result_df5$number_of_patients)

number_of_patients<-apply(number_of_patients,1,mean)
min_accuracy<-apply(accuracy,1,min)
max_accuracy<-apply(accuracy,1,max)
mean_accuracy<-apply(accuracy,1,mean)

df<-data.frame(min_accuracy=min_accuracy,mean_accuracy=mean_accuracy,max_accuracy=max_accuracy,number_of_patients=number_of_patients)
row.names(df)<-common_name
df$number_of_patients<-as.integer(df$number_of_patients)
df<-df[rev(order(df$number_of_patients)),]

write.csv(df,"logistic_train_accuracy.csv")
