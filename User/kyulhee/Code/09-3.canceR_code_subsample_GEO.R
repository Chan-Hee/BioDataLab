setwd("/home/tjahn/Data")

file_name <- "FinalData_GSM_gene_index_result_without_rare_cancer.csv"
df <- read.csv(file_name,header = TRUE)
#gene_set <- read.csv("C:\\test\\sam\\inter_GEO_TCGA_geneset_final.csv", header = FALSE)
print(paste0("***************************  file reads."))
gene_set <- read.csv("/home/tjahn/Git2/User/kyulhee/TCGA/inter_GEO_TCGA_geneset_final.csv", header = FALSE)
gene_set <- as.character(gene_set$V1)

patients <- df$X
result <- df$result
index <- df$index
cancer_code <- df$cancer_code

# Modifying data frame
names(df) <- gsub(".", "-", names(df), fixed = TRUE)
print(paste0("***************************  modified."))

# Subsampling
df_sub <- subset(df, select = gene_set)
 
final_csv <- cbind(patients, df_sub, result, cancer_code, index)
write.csv(final_csv,"wt_cancer_code_FinalData_GSM_gene_index_result_without_rare_cancer_4964.csv",row.names = FALSE)
print(paste0("***************************  subsampled."))