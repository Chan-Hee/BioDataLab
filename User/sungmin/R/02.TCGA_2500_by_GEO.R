#setwd("/home/tjahn/Data/")
forms <- c("FPKM-UQ","FPKM","htseq")
#input <- "/home/tjahn/Data/01.4964/"
#output<- "/home/tjahn/Data/sungminTCGAtoGEO_2500/"
for(form in forms){
  
  df<-read.csv("/home/tjahn/Data/01.4964/inter_wt_cancer_code_FinalData_GSM_gene_index_result_without_rare_cancer_4964.csv", header=TRUE)
  names(df) <- gsub(".", "-", names(df), fixed = TRUE)
  print(paste0("large table : ", form, " reads."))
  
  TCGA_2500 <-read.csv(paste0("/home/tjahn/Data/sungminTCGAtoGEO_2500/TCGA_",form,"_2500_names.csv"))
  TCGA_2500 <-as.character(TCGA_2500$x)
  TCGA_2500 <-gsub(".", "-", TCGA_2500, fixed = TRUE)
  #GEO_2500 <- read.csv("GEO_2500_names.csv", header = TRUE)
  #GEO_2500 <- as.character(GEO_2500$x)
  #GEO_2500 <-  gsub(".", "-", GEO_2500, fixed = TRUE)
  
  genes <- df[,2:(ncol(df)-3)]
  patients <- df$patient
  result <- df$result
  cancer_code <- df$cancer_code
  index <- df$index
  genes <- subset(genes, select = TCGA_2500)
  df_ch <- cbind(patients, genes, result, cancer_code, index)
  print("df modified.")

  write.csv(df_ch, paste0("/home/tjahn/Data/sungminTCGAtoGEO_2500/inter_wt_cancer_code_FinalData_GSM_gene_index_result_without_rare_cancer_",form,"_2500.csv"), row.names=FALSE)
  print(paste0("modified table: ", form, " written."))
}