#setwd("E:\\Lab\\TCGA_data")
setwd("/home/tjahn/GDC_Data/GeneExpression/Gene_txt_sum/Final")
df<-read.csv("Final_TCGA_gene_expression_FPKM-UQ_4964_ch_reduced.csv")
df_ch <- df[sample(nrow(df)),]

patient <- df_ch$patient
result<-df_ch$result
gene_expression<-df_ch[,2:4965]
names(gene_expression) <- gsub(".", "-", names(gene_expression), fixed = TRUE)

remain <- length(df_ch$result)%%5
index<-rep(1:5,length(df_ch$result)/5)
if(remain>0){
  index <- c(index, 1:remain)
}

data<-cbind(patient,gene_expression,index,result)

write.csv(data,"Final_TCGA_gene_expression_FPKM-UQ_4964_ch_reduced_final.csv", row.names = FALSE)
