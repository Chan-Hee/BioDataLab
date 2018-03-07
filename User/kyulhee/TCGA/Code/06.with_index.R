#setwd("C:\\test\\sam")
setwd("/home/tjahn/GDC_Data/GeneExpression/Gene_txt_sum/Final")
df<-read.csv("Final_TCGA_gene_expression_FPKM-UQ_4964.csv")
result<-df$result
gene_expression<-round(df[,2:4965],3)
names(gene_expression) <- gsub(".", "-", names(gene_expression), fixed = TRUE)
patient<-df$file_name

remain <- length(df$result)%%5
index <- rep(1:5,length(df$result)/5)
if(remain>0){
  index <- c(index, 1:remain)
}

data<-cbind(patient,gene_expression,index,result)

write.csv(data,"Final_TCGA_gene_expression_FPKM-UQ_4964_ch.csv", row.names = FALSE)