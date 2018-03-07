#################################################### VarianceTest ##########################################
GetVar<-function(genes){
  VAR<-apply(genes,2,sd)
  return(VAR)
}

#################################################### Main ##################################################
#setwd("/home/tjahn/Data/")
df<-read.csv("/home/tjahn/Data/01.4964/inter_wt_cancer_code_Final_TCGA_gene_expression_FPKM_4964.csv")
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

write.csv(df_ch, "/home/tjahn/Data/sungminTCGAtoGEO_2500/inter_wt_cancer_code_Final_TCGA_gene_expression_FPKM_2500.csv", row.names=FALSE)
write.csv(names(genes), "/home/tjahn/Data/sungminTCGAtoGEO_2500/TCGA_FPKM_2500_names.csv", row.names=FALSE)

#################
df<-read.csv("/home/tjahn/Data/01.4964/inter_wt_cancer_code_Final_TCGA_gene_expression_FPKM-UQ_4964.csv")
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

write.csv(df_ch, "/home/tjahn/Data/sungminTCGAtoGEO_2500/inter_wt_cancer_code_Final_TCGA_gene_expression_FPKM-UQ_2500.csv", row.names=FALSE)
write.csv(names(genes), "/home/tjahn/Data/sungminTCGAtoGEO_2500/TCGA_FPKM-UQ_2500_names.csv", row.names=FALSE)
###############
df<-read.csv("/home/tjahn/Data/01.4964/inter_wt_cancer_code_Final_TCGA_gene_expression_htseq_4964.csv")
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

write.csv(df_ch, "/home/tjahn/Data/sungminTCGAtoGEO_2500/TCGA_htseq_2500_names.csv", row.names=FALSE)
write.csv(names(genes), "GEO_2500_names.csv", row.names=FALSE)

