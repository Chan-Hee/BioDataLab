setwd("/home/tjahn/Data")

file_name <- "FinalData_GSM_gene_index_result_without_rare_cancer.csv"
large_table <- read.csv(file_name,header = TRUE)
#gene_set <- read.csv("C:\\test\\sam\\inter_GEO_TCGA_geneset_final.csv", header = FALSE)
print(paste0("***************************  file reads."))
gene_set <- read.csv("/home/tjahn/Git2/User/kyulhee/TCGA/inter_GEO_TCGA_geneset_final.csv", header = FALSE)
gene_set <- as.character(gene_set$V1)

patients <- large_table$X
result <- large_table$result
index <- large_table$index
cancer_code <- large_table$cancer_code

# Modifying large table
names(large_table) <- gsub(".", "-", names(large_table), fixed = TRUE)
print(paste0("***************************  modified."))

# Subsampling
sub_table <- subset(large_table, select = gene_set)
 
len <- length(large_table)
final_csv <- cbind(patients, sub_table, result, cancer_code, index)
write.csv(final_csv,"wt_cancer_code_FinalData_GSM_gene_index_result_without_rare_cancer_4964.csv",row.names = FALSE)
print(paste0("***************************  subsampled."))