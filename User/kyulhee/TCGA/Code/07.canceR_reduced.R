set.seed(777)
#setwd("C:\\test\\sam")
setwd("/home/tjahn/GDC_Data/GeneExpression/Gene_txt_sum/Final")
df<-read.csv("Final_TCGA_gene_expression_htseq_4964_ch.csv")

#result<-c(0,1,0,1,1,1)
#data <- c("a", "b", "c", "d", "e", "f")
#df <- data.frame(data, result)

names(df) <- gsub(".", "-", names(df), fixed = TRUE)
cancer <- df[df$result==1,]
normal <- df[df$result==0,]

sample <- cancer[sample(length(cancer[,1]), length(normal[,1])),]

data <- rbind(normal, sample)

write.csv(data,"Final_TCGA_gene_expression_htseq_4964_ch_reduced.csv", row.names = FALSE)