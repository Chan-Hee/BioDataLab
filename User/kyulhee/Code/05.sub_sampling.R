#setwd("C:\\test\\sam")
setwd("/home/tjahn/GDC_Data/GeneExpression/Gene_txt_sum/Final")
#large_table <- read.csv("TCGA_genes_FPKM-UQ_1.csv",header = TRUE)
#forms <- c("FPKM-UQ", "FPKM", "htseq")
forms <- c("htseq")
print("***************************  file table done.")

for(form in forms){
  
  file_name <- paste0("Final_TCGA_gene_expression_",form,".csv")
  print(paste0("***************************  format: @ ", form, " @ starts."))
  
  large_table <- read.csv(file_name,header = TRUE)
  #gene_set <- read.csv("C:\\test\\sam\\inter_GEO_TCGA_geneset_final.csv", header = FALSE)
  print(paste0("***************************  format: @ ", form, " @ reads."))
  gene_set <- read.csv("/home/tjahn/Git2/User/kyulhee/TCGA/inter_GEO_TCGA_geneset_final.csv", header = FALSE)
  gene_set <- as.character(gene_set$V1)

  # Modifying large table
  names(large_table) <- gsub(".", "-", names(large_table), fixed = TRUE)
  write.csv(large_table, "Final_TCGA_gene_expression_FPKM_Modified.csv",row.names = FALSE)
  print(paste0("***************************  ", form, " @ modified."))
  
  # Subsampling
  sub_table <- subset(large_table, select = gene_set)

  len <- length(large_table)
  final_csv <- cbind(large_table[1],sub_table,large_table[len-2], large_table[len-1], large_table[len])
  write.csv(final_csv,paste0("Final_TCGA_gene_expression_",form,"_4964.csv"),row.names = FALSE)
  print(paste0("***************************  format: @ ", form, " @ subsampled."))
}