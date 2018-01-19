weighted_sum0 <- read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/DNN/result/result_weigthed_sum0.csv")
names(weighted_sum0)[1]<-"index"

weighted_sum1 <- read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/DNN/result/result_weigthed_sum1.csv")
names(weighted_sum1)[1]<-"index"

weighted_sum2 <- read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/DNN/result/result_weigthed_sum2.csv")
names(weighted_sum2)[1]<-"index"

weighted_sum3 <- read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/DNN/result/result_weigthed_sum3.csv")
names(weighted_sum3)[1]<-"index"

weighted_sum4 <- read.csv("/Users/chanhee/Desktop/BioDataLab/User/chanhee/DNN/result/result_weigthed_sum4.csv")
names(weighted_sum4)[1]<-"index"

index_genes<-weighted_sum0[,1:2]
sum_of_weights<-abs(as.matrix(weighted_sum0[,3:4]))+abs(as.matrix(weighted_sum1[,3:4]))+abs(as.matrix(weighted_sum2[,3:4]))+abs(as.matrix(weighted_sum3[,3:4]))+abs(as.matrix(weighted_sum4[,3:4]))

weighted_sum<-cbind(index_genes,sum_of_weights)
weighted_sum$abs_sum<-abs(weighted_sum$weighted_sum)

weighted_sum<-weighted_sum[rev(order(weighted_sum$abs_sum)),]
weighted_sum$gene_names<-as.character(weighted_sum$gene_names)

data<-read.csv("/Users/chanhee/Desktop/FinalData_GSM_gene_index_result.csv")
row.names(data)<-data$X
data<-data[,-1]
selected_gene_index<-weighted_sum$index+1

data10off<-data[,c(selected_gene_index[601:6000],6001,6002)]
data20off<-data[,c(selected_gene_index[1201:6000],6001,6002)]
data30off<-data[,c(selected_gene_index[1801:6000],6001,6002)]
data40off<-data[,c(selected_gene_index[2401:6000],6001,6002)]
data50off<-data[,c(selected_gene_index[3001:6000],6001,6002)]
data60off<-data[,c(selected_gene_index[3601:6000],6001,6002)]
data70off<-data[,c(selected_gene_index[4201:6000],6001,6002)]
data80off<-data[,c(selected_gene_index[4801:6000],6001,6002)]
data90off<-data[,c(selected_gene_index[5401:6000],6001,6002)]
data95off<-data[,c(selected_gene_index[5701:6000],6001,6002)]
data99off<-data[,c(selected_gene_index[5941:6000],6001,6002)]


write.csv(data10off,"/Users/chanhee/Desktop/FinalData10off_GSM_gene_index_result.csv")
write.csv(data20off,"/Users/chanhee/Desktop/FinalData20off_GSM_gene_index_result.csv")
write.csv(data30off,"/Users/chanhee/Desktop/FinalData30off_GSM_gene_index_result.csv")
write.csv(data40off,"/Users/chanhee/Desktop/FinalData40off_GSM_gene_index_result.csv")
write.csv(data50off,"/Users/chanhee/Desktop/FinalData50off_GSM_gene_index_result.csv")
write.csv(data60off,"/Users/chanhee/Desktop/FinalData60off_GSM_gene_index_result.csv")
write.csv(data70off,"/Users/chanhee/Desktop/FinalData70off_GSM_gene_index_result.csv")
write.csv(data80off,"/Users/chanhee/Desktop/FinalData80off_GSM_gene_index_result.csv")
write.csv(data90off,"/Users/chanhee/Desktop/FinalData90off_GSM_gene_index_result.csv")
write.csv(data95off,"/Users/chanhee/Desktop/FinalData95off_GSM_gene_index_result.csv")
write.csv(data99off,"/Users/chanhee/Desktop/FinalData99off_GSM_gene_index_result.csv")


