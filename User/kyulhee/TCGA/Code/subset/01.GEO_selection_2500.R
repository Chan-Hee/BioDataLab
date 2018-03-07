#################################################### VarianceTest ##########################################
GetVar<-function(genes){
  VAR<-apply(genes,2,sd)
  return(VAR)
}

#################################################### Main ##################################################
setwd("/home/tjahn/Data/")
df<-read.csv("inter_wt_cancer_code_FinalData_GSM_gene_index_result_without_rare_cancer_4964.csv")
names(df) <- gsub(".", "-", names(df), fixed = TRUE)
genes <- df[,2:(ncol(df)-3)]
patients <- df$patient
result <- df$result
cancer_code <- df$cancer_code
index <- df$index

VAR <- GetVar(genes)
genes <- rbind(genes,VAR)
genes <- genes[,rev(order(genes[nrow(genes),]))]
genes <- genes[-nrow(genes),1:2500]
df_ch <- cbind(patient, genes, result, cancer_code, index)

write.csv(df_ch, "inter_wt_cancer_code_FinalData_GSM_gene_index_result_without_rare_cancer_2500.csv", row.names=FALSE)
write.csv(names(genes), "GEO_2500_names.csv", row.names=FALSE)
