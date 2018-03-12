#setwd("C:\\test\\sam")
setwd("/home/tjahn/GDC_Data/GeneExpression/Gene_txt_sum/Final/01.merge")
df<-read.csv("Final_TCGA_gene_expression_htseq.csv")

gene_expression<-round(df[,2:15830],3)
names(gene_expression) <- gsub(".", "-", names(gene_expression), fixed = TRUE)
patient<-df$file_name
barcode <- df$barcode
result<-df$result

ref <- read.csv("/home/tjahn/Git2/User/kyulhee/TCGA/TCGA_GEO_pro.csv", header = TRUE, na.strings = "NNN")

for(j in 1:nrow(df)){
  cancer_code <- c(cancer_code, as.character(ref[ref[,1] == unlist(strsplit(as.character(df[j,barcode]), "-"))[2],3]))
}
df_ch <- cbind(patient, gene_expression, cancer_code, result)

remain <- nrow(df)%%5
index <- rep(1:5,nrow(df)/5)
if(remain>0){
  index <- c(index, 1:remain)
}

data<-cbind(patient,gene_expression,index,result)

write.csv(data,"Final_TCGA_gene_expression_htseq_ch.csv", row.names = FALSE)