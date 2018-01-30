getwd()

Before_data <- read.csv("C:/Users/bibs-student/Desktop/¼º¹Î_DNN/FinalData_GSM_gene_index_result.csv",header = T, sep = ",",row.names = 1)
Before_data <- Before_data[,-which(names(Before_data)%in%"index")]
CancerCode <-read.csv("C:/Users/bibs-student/Desktop/¼º¹Î_DNN/GPL570_sampleinfo.txt",sep = "\t",header = T)
GSM <- row.names(Before_data)
row.names(CancerCode) <-CancerCode$GSM_ID
CancerCode<-CancerCode[GSM,]
all(row.names(CancerCode) == row.names(Before_data))
Before_data$cancer_code <- CancerCode$CANCER_CODE

count_cancer<-as.data.frame(table(Before_data$cancer_code))
count_cancer <-count_cancer[count_cancer$Freq>15,]
freq_cancer <- count_cancer$Var1


After_data <- Before_data[which(Before_data$cancer_code %in% freq_cancer),]
dim(After_data)
dim(Before_data)

count = nrow(After_data)/5
count = as.integer(count)
remainder = nrow(After_data)%%5

if(remainder!=0){
  Index <- c(rep(1:5,count),1:remainder)
}else{
  Index <-rep(1:5,count)}
After_data$index <- Index

write.csv(After_data,"C:/Users/bibs-student/Desktop/¼º¹Î_DNN/FinalData_GSM_gene_index_result_without_rare_cancer.csv")
