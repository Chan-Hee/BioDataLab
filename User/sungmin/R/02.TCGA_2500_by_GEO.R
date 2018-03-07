setwd("/home/tjahn/Data/")
forms <- c("FPKM-UQ","FPKM","htseq")

for(form in forms){
  
  df<-read.csv(paste0("inter_wt_cancer_code_Final_TCGA_gene_expression_",form,"_4964.csv"), header=TRUE)
  names(df) <- gsub(".", "-", names(df), fixed = TRUE)
  print(paste0("large table : ", form, " reads."))
  GEO_2500 <- read.csv("GEO_2500_names.csv", header = TRUE)
  GEO_2500 <- as.character(GEO_2500$x)
  GEO_2500 <-  gsub(".", "-", GEO_2500, fixed = TRUE)
  
  genes <- df[,2:(ncol(df)-3)]
  patients <- df$patient
  result <- df$result
  cancer_code <- df$cancer_code
  index <- df$index
  genes <- subset(genes, select = GEO_2500)
  df_ch <- cbind(patients, genes, result, cancer_code, index)
  print("df modified.")

  write.csv(df_ch, paste0("inter_wt_cancer_code_Final_TCGA_gene_expression_",form,"_2500.csv"), row.names=FALSE)
  print(paste0("modified table: ", form, " written."))
}